import numpy as np
from typing import Dict
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Independent, Categorical

from aevnmt.dist import get_named_params

from .generative import GenerativeLM, GenerativeTM
from .inference import InferenceModel

from probabll.distributions import MixtureOfGaussians


class AEVNMT(nn.Module):

    def __init__(self, latent_size, src_embedder, tgt_embedder,
            language_model: GenerativeLM, translation_model: GenerativeTM, inference_model: InferenceModel,
            dropout, tied_embeddings, prior_family: str, prior_params: list,
            feed_z=False,
            aux_lms: Dict[str, GenerativeLM]=dict(), aux_tms: Dict[str, GenerativeTM]=dict(),
            mixture_likelihood=False, mixture_likelihood_dir_prior=0.0,
            mdr=False, lag_side=None):
        super().__init__()
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.latent_size = latent_size
        self.language_model = language_model
        self.translation_model = translation_model
        self.inference_model = inference_model

        self.mixture_likelihood = mixture_likelihood
        self.mixture_likelihood_dir_prior = mixture_likelihood_dir_prior
        # Auxiliary LMs and TMs
        self.aux_lms = nn.ModuleDict(aux_lms)
        self.aux_tms = nn.ModuleDict(aux_tms)

        if mdr:
            self.lag = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(1, 1),
                torch.nn.Softplus()
            )
        else:
            self.lag = None

        if lag_side is not None:
            self.lag_side= torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(1, 1),
                torch.nn.Softplus()
            )
            self.lag_side_target=lag_side
        else:
            self.lag_side= None
            self.lag_side_target=None

        # This is done because the location and scale of the prior distribution are not considered
        # parameters, but are rather constant. Registering them as buffers still makes sure that
        # they will be moved to the appropriate device on which the model is run.
        self.prior_family = prior_family
        if prior_family == "gaussian":
            if not prior_params:
                prior_params = [0., 1.]
            if len(prior_params) != 2:
                raise ValueError("Specify the Gaussian prior using a location and a strictly positive scale.")
            if prior_params[1] <= 0:
                raise ValueError("The scale of a Gaussian distribution is strictly positive: %r" % prior_params)
            self.register_buffer("prior_loc", torch.zeros([latent_size]))
            self.register_buffer("prior_scale", torch.ones([latent_size]))
        elif prior_family == "beta":
            if not prior_params:
                prior_params = [0.5, 0.5]
            if len(prior_params) != 2:
                raise ValueError("Specify the Beta prior using two strictly positive shape parameters.")
            if prior_params[0] <= 0. or prior_params[1] <= 0.:
                raise ValueError("The shape parameters of a Beta distribution are strictly positive: %r" % prior_params)
            self.register_buffer("prior_a", torch.full([latent_size], prior_params[0]))
            self.register_buffer("prior_b", torch.full([latent_size], prior_params[1]))
            if inference_model.family != "kumaraswamy":
                raise ValueError("I think you forgot to change your posterior distribution to something with support (0,1)")
        elif prior_family == "mog":
            if not prior_params:
                prior_params = [10, 10, 0.5]
            if len(prior_params) != 3:
                raise ValueError("Specify the MoG prior using a number of components, a radius (for initialisation), and a strictly positive scale.")
            num_components = int(prior_params[0])
            if num_components <= 1:
                raise ValueError("An MoG prior requires more than 1 component.")
            prior_radius = prior_params[1]
            if prior_radius <= 0:
                raise ValueError("Initialising the MoG prior takes a strictly positive radius.")
            prior_scale = prior_params[2]
            if prior_scale <= 0:
                raise ValueError("The prior variance must be strictly positive.")
            # uniform prior over components
            self.register_buffer("prior_logits", torch.ones(num_components))
            self.register_buffer("prior_locations", - prior_radius + torch.rand([num_components, latent_size]) * 2 * prior_radius )
            self.register_buffer("prior_scales", torch.full([num_components, latent_size], prior_scale))
        else:
            raise NotImplementedError("I cannot impose a %s prior on the latent code." % prior_family)

    def lagrangian_multiplier(self, device):
        if self.lag is None:
            raise ValueError("You are not using MDR, thus there's no Lagrangian multiplier")
        else:
            return self.lag(torch.zeros(1, device=device))

    def mdr_parameters(self):
        if self.lag is None:
            return []
        else:
            return self.lag.parameters()

    def lagrangian_multiplier_side(self, device):
        if self.lag_side is None:
            raise ValueError("You are not using Lagrangian side losses target, thus there's no Lagrangian multiplier")
        else:
            return self.lag_side(torch.zeros(1, device=device))

    def lag_side_parameters(self):
        if self.lag_side is None:
            return []
        else:
            return self.lag_side.parameters()


    def inference_parameters(self):
        return self.inference_model.parameters()

    def embedding_parameters(self):
        return chain(self.src_embedder.parameters(), self.tgt_embedder.parameters())

    def generative_parameters(self):
        return chain(self.lm_parameters(), self.tm_parameters(), self.aux_lm_parameters(), self.aux_tm_parameters())

    def aux_lm_parameters(self):
        return chain(*[model.parameters() for model in self.aux_lms.values()])

    def aux_tm_parameters(self):
        return chain(*[model.parameters() for model in self.aux_tms.values()])

    def lm_parameters(self):
        return chain(self.src_embedder.parameters(), self.language_model.parameters())

    def tm_parameters(self):
        if self.tgt_embedder is not None:
            return chain(self.tgt_embedder.parameters(), self.translation_model.parameters())
        else:
            return iter(())

    def approximate_posterior(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        """
        Returns an approximate posterior distribution q(z|x, y).
        """
        return self.inference_model(x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y)

    def prior(self) -> Distribution:
        if self.prior_family == "gaussian":
            p = Independent(torch.distributions.Normal(loc=self.prior_loc, scale=self.prior_scale), 1)
        elif self.prior_family == "beta":
            p = Independent(torch.distributions.Beta(self.prior_a, self.prior_b), 1)
        elif self.prior_family == "mog":
            p = MixtureOfGaussians(logits=self.prior_logits, locations=self.prior_locations, scales=self.prior_scales)
        return p

    def src_embed(self, x):
        x_embed = self.src_embedder(x)
        x_embed = self.dropout_layer(x_embed)
        return x_embed

    def tgt_embed(self, y):
        y_embed = self.tgt_embedder(y)
        y_embed = self.dropout_layer(y_embed)
        return y_embed

    def forward(self, x, seq_mask_x, seq_len_x, y, x_shuf,seq_mask_x_shuf,seq_len_x_shuf,y_shuf,seq_mask_y_shuf,seq_len_y_shuf, z):
        """
        Run all components of the model and return parameterised distributions.

        Returns:
            tm_likelihood: Categorical distributions Y_j|x,z, y_{<j} with shape [B, Ty, Vy]
            lm_likelihood: Categorical distributions X_i|z,x_{<i} with shape [B, Tx, Vx]
            state: a dictionary with information from decoders' forward passes (e.g. `att_weights`)
            aux_lm_likelihoods: dictionary mapping auxiliary LMs to their parameterised distributions
            aux_tm_likelihoods: dictionary mapping auxiliary TMs to their parameterised distributions
        """
        state = dict()
        lm_likelihood = self.language_model(x, z, state)
        if y is not None:
            tm_likelihood = self.translation_model(x, seq_mask_x, seq_len_x, y, z, state)
        else:
            tm_likelihood=None

        # Obtain auxiliary X_i|z, x_{<i}
        aux_lm_likelihoods = dict()
        for aux_name, aux_decoder in self.aux_lms.items():
            if aux_name == "shuffled":
                if x_shuf is not None:
                    aux_lm_likelihoods[aux_name] = aux_decoder(x_shuf, z)
            else:
                aux_lm_likelihoods[aux_name] = aux_decoder(x, z)


        # Obtain auxiliary Y_j|z, x, y_{<j}
        aux_tm_likelihoods = dict()
        if y is not None:
            for aux_name, aux_decoder in self.aux_tms.items():
                if aux_name == "shuffled":
                    #Xs are ignored
                    if y_shuf is not None:
                        aux_tm_likelihoods[aux_name] = aux_decoder(x, seq_mask_x, seq_len_x, y_shuf, z)
                else:
                    aux_tm_likelihoods[aux_name] = aux_decoder(x, seq_mask_x, seq_len_x, y, z)

        return tm_likelihood, lm_likelihood, state, aux_lm_likelihoods, aux_tm_likelihoods

    def log_likelihood_tm(self, comp_name, likelihood: Distribution, y):
        return self.aux_tms[comp_name].log_prob(likelihood, y)

    def log_likelihood_lm(self, comp_name, likelihood: Distribution, x, per_token_norm=False):
        if per_token_norm:
            return self.aux_lms[comp_name].log_prob_norm(likelihood, x)
        else:
            return self.aux_lms[comp_name].log_prob(likelihood, x)

    def loss(self, tm_likelihood: Categorical, lm_likelihood: Categorical, targets_y, targets_x, targets_y_shuf, targets_x_shuf, qz: Distribution,
            free_nats=0., KL_weight=1., reduction="mean", aux_lm_likelihoods=dict(), aux_tm_likelihoods=dict(),disable_main_loss=False, disable_side_losses=False):
        """
        Computes an estimate of the negative evidence lower bound for the single sample of the latent
        variable that was used to compute the categorical parameters, and the distributions qz
        that the sample comes from.

        :param tm_likelihood: Categorical distributions from LM with shape [B, Ty, Vy]
        :param lm_likelihood: Categorical distributions from TM with shape [B, Tx, Vx]
        :param targets_y: target labels target sentence [B, T_y]
        :param targets_x: target labels source sentence [B, T_x]
        :param qz: distribution that was used to sample the latent variable.
        :param free_nats: KL = min(free_nats, KL)
        :param KL_weight: weight to multiply the KL with, applied after free_nats
        :param reduction: what reduction to apply, none ([B]), mean ([]) or sum ([])
        :param aux_lm_likelihoods: a dictionary with LM likelihoods
        :param aux_tm_likelihoods: a dictionary with TM likelihoods
        """
        # [B]
        tm_log_likelihood=0.0
        if tm_likelihood is not None:
            tm_log_likelihood = self.translation_model.log_prob(tm_likelihood, targets_y)
        tm_loss = - tm_log_likelihood

        # [B]
        lm_log_likelihood=0.0
        if lm_likelihood is not None:
            lm_log_likelihood = self.language_model.log_prob(lm_likelihood, targets_x)
        lm_loss = - lm_log_likelihood


        # Compute the KL divergence between the distribution used to sample z, and the prior
        # distribution.
        pz = self.prior()

        out_dict = dict()
        # TODO: N this is [...,D], whereas with MoG this is [...]
        #  we need to wrap stuff around torch.distributions.Independent
        KL = torch.distributions.kl.kl_divergence(qz, pz)
        raw_KL = KL * 1
        mdr_term = 0.
        mdr_loss = 0.
        if free_nats > 0:
            if self.lag is None:
                KL = torch.clamp(KL, min=free_nats)
            else:
                u = self.lagrangian_multiplier(KL.device)
                rate = KL.mean()
                mdr_term = u.detach() * (free_nats - rate)
                mdr_loss = - u * (free_nats - rate.detach())
                out_dict['mdr_loss'] = mdr_loss
        # annealing
        KL *= KL_weight

        out_dict['KL'] = KL
        out_dict['raw_KL'] = raw_KL

        # Alternative views of p(x|z)
        # [Cx, B]
        side_lm_likelihood = torch.zeros([len(aux_lm_likelihoods), KL.size(0)], dtype=KL.dtype, device=KL.device)
        side_lm_likelihood_per_token_norm = torch.zeros([len(aux_lm_likelihoods), KL.size(0)], dtype=KL.dtype, device=KL.device)
        for c, (aux_name, aux_likelihood) in enumerate(aux_lm_likelihoods.items()):
            out_dict['lm/' + aux_name] = self.log_likelihood_lm(aux_name, aux_likelihood, targets_x_shuf if aux_name == "shuffled" else targets_x )
            if aux_name == "shuffled":
                #Create a new loss normalized by number of tokens
                out_dict['lm/' + aux_name + '_normtok'] = self.log_likelihood_lm(aux_name, aux_likelihood, targets_x_shuf if aux_name == "shuffled" else targets_x ,per_token_norm=True)
            else:
                out_dict['lm/' + aux_name + '_normtok'] = out_dict['lm/' + aux_name]

            w=1.0
            if hasattr(self.aux_lms[aux_name],'normalize_weight') and self.aux_lms[aux_name].normalize_weight:
                w=targets_x.size(1)/(1.0*self.aux_lms[aux_name].vocab_size)
            side_lm_likelihood[c] = out_dict['lm/' + aux_name]
            side_lm_likelihood_per_token_norm[c] = out_dict['lm/' + aux_name + '_normtok']

        # Alternative views of p(y|z,x)
        # [Cy, B]

        side_tm_likelihood = torch.zeros([len(aux_tm_likelihoods), KL.size(0)], dtype=KL.dtype, device=KL.device)
        if targets_y is not None:
            for c, (aux_name, aux_likelihood) in enumerate(aux_tm_likelihoods.items()):
                out_dict['tm/' + aux_name] = self.log_likelihood_tm(aux_name, aux_likelihood, targets_y_shuf if aux_name == "shuffled" else targets_y)
                w=1.0
                if hasattr(self.aux_tms[aux_name],'normalize_weight') and self.aux_tms[aux_name].normalize_weight:
                    w=targets_y.size(1)/(1.0*self.aux_tms[aux_name].vocab_size)
                side_tm_likelihood[c] = out_dict['tm/' + aux_name]*w

        if not self.mixture_likelihood:
            # ELBO
            #  E_q[ \log P(x|z,c=main) P(y|z,x,c=main)] - KL(q(z) || p(z))
            #  + E_q[\sum_{c not main} log P(x|z,c) + log P(y|z,x,c) ]
            # where the second row are heuristic side losses (we can think of it as multitask learning)
            elbo = tm_log_likelihood + lm_log_likelihood - KL
            # we sum the alternative views (as in multitask learning)
            aux_log_likelihood = side_lm_likelihood.sum(0) + side_tm_likelihood.sum(0)
            aux_log_likelihood_per_token_norm = side_lm_likelihood_per_token_norm.sum(0)
            if disable_side_losses:
                aux_log_likelihood=0.0
            if disable_main_loss:
                elbo = - KL

            side_elbo=aux_log_likelihood - KL
            side_elbo_per_token_norm=aux_log_likelihood_per_token_norm - KL
            lag_side_loss=0.0
            lag_side_term=0.0
            #Lagrangian multiplier if needed
            if self.lag_side is not None:
                u = self.lagrangian_multiplier_side(aux_log_likelihood.device)
                rate = -side_elbo.mean()
                lag_side_term = u.detach() * (rate -  self.lag_side_target )
                #If current neg. side ELBO > target neg. side ELBO, constraint
                #has not been met. Difference is positive, in order to
                #minimize lag_side_loss, u should be big

                #If current neg. side ELBO < target neg. side ELBO, constraint
                #has been met. Difference is negative, in order to minimize
                #lag_side_loss, u should be zero
                lag_side_loss = - u * ( rate.detach() - self.lag_side_target)
                out_dict['lag_side_loss'] = lag_side_loss
                out_dict['lag_difference']=rate.detach() - self.lag_side_target
                loss = - (elbo - lag_side_term - mdr_term)
            else:
                loss = - (elbo + aux_log_likelihood - mdr_term)

            # main log-likelihoods
            out_dict['lm/main'] = lm_log_likelihood
            out_dict['tm/main'] = tm_log_likelihood
            out_dict['side'] = aux_log_likelihood
            out_dict['side_per_token_norm'] = aux_log_likelihood_per_token_norm
            out_dict['ELBO']=elbo
            out_dict['sideELBO']=side_elbo
            out_dict['sideELBO_per_token_norm']=side_elbo_per_token_norm
        else:
            assert disable_main_loss == False
            assert disable_side_losses == False
            # ELBO uses mixture models for X|z and Y|z,x:
            #  E_q[ \log P(x|z) + \log P(y|z,x)] - KL(q(z) || p(z))
            #   where \log P(x|z)   = \log \sum_{c=1}^{Cy} w_c P(x|z,c)
            #   and   \log P(y|z,x) = \log \sum_{c=1}^{Cx} w_c P(y|z,x,c)
            Cx = len(aux_lm_likelihoods) + 1
            if self.mixture_likelihood_dir_prior == 0:
                wx = torch.full([KL.size(0), Cx], 1. / Cx, dtype=KL.dtype, device=KL.device).permute(1, 0)
            else:
                wx = torch.distributions.Dirichlet(
                    torch.full([KL.size(0), Cx], self.mixture_likelihood_dir_prior,
                        dtype=KL.dtype, device=KL.device)).sample().permute(1, 0)
            # [Cx, B] -> [B]
            lm_mixture = (torch.cat([lm_log_likelihood.unsqueeze(0), side_lm_likelihood]) - torch.log(wx)).logsumexp(0)
            Cy = len(aux_tm_likelihoods) + 1
            if self.mixture_likelihood_dir_prior == 0:
                wy = torch.full([KL.size(0), Cy], 1. / Cy, dtype=KL.dtype, device=KL.device).permute(1, 0)
            else:
                wy = torch.distributions.Dirichlet(
                    torch.full([KL.size(0), Cy], self.mixture_likelihood_dir_prior,
                        dtype=KL.dtype, device=KL.device)).sample().permute(1, 0)
            # [Cy, B] -> [B]
            tm_mixture = (torch.cat([tm_log_likelihood.unsqueeze(0), side_tm_likelihood]) - torch.log(wy)).logsumexp(0)
            elbo = lm_mixture + tm_mixture - KL
            loss = - elbo
            out_dict['lm/main'] = lm_mixture
            out_dict['tm/main'] = tm_mixture
            out_dict['lm/recurrent'] = lm_log_likelihood
            out_dict['tm/recurrent'] = tm_log_likelihood
            out_dict['ELBO']=elbo

        # Return differently according to the reduction setting.
        if reduction == "mean":
            out_dict['loss'] = loss.mean()
        elif reduction == "sum":
            out_dict['loss'] = loss.sum()
        elif reduction == "none":
            out_dict['loss'] = loss
        else:
            raise Exception(f"Unknown reduction option {reduction}")

        return out_dict
