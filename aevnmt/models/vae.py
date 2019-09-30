import torch
import torch.nn as nn
import torch.nn.functional as F

from aevnmt.components import RNNEncoder, tile_rnn_hidden, tile_rnn_hidden_for_decoder
from aevnmt.dist import NormalLayer

from itertools import chain

class InferenceNetwork(nn.Module):

    def __init__(self, src_embedder, hidden_size, latent_size, bidirectional, num_enc_layers, cell_type,max_pool):
        """
        :param src_embedder: uses this embedder, but detaches its output from the graph as to not compute
                             gradients for it.
        """
        super().__init__()
        self.max_pool=max_pool

        self.src_embedder = src_embedder
        emb_size = src_embedder.embedding_dim
        self.encoder = RNNEncoder(emb_size=emb_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  dropout=0.,
                                  num_layers=num_enc_layers,
                                  cell_type=cell_type)
        encoding_size = hidden_size if not bidirectional else hidden_size * 2
        self.normal_layer = NormalLayer(encoding_size, hidden_size, latent_size)

    def forward(self, x, seq_mask_x, seq_len_x):
        x_embed = self.src_embedder(x).detach()
        encoder_outputs, _ = self.encoder(x_embed, seq_len_x) #(B, T, hidden_size)

        if self.max_pool:
            avg_encoder_output = encoder_outputs.max(dim=1)[0]
        else:
            avg_encoder_output = (encoder_outputs * seq_mask_x.unsqueeze(-1).type_as(encoder_outputs)).sum(dim=1)

        return self.normal_layer(avg_encoder_output)

    def parameters(self, recurse=True):
        return chain(self.encoder.parameters(recurse=recurse), self.normal_layer.parameters(recurse=recurse))

    def named_parameters(self, prefix='', recurse=True):
        return chain(self.encoder.named_parameters(prefix='', recurse=True), self.normal_layer.named_parameters(prefix='', recurse=True), )

class BilingualInferenceNetwork(nn.Module):

    def __init__(self, src_embedder, tgt_embedder, hidden_size, latent_size, bidirectional, num_enc_layers, cell_type, max_pool):
        """
        :param src_embedder: uses this embedder, but detaches its output from the graph as to not compute
                             gradients for it.
        :param tgt_embedder: uses this embedder, but detaches its output from the graph as to not compute
                             gradients for it.
        """
        super().__init__()
        self.max_pool=max_pool

        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        emb_size = src_embedder.embedding_dim
        self.src_encoder = RNNEncoder(emb_size=emb_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  dropout=0.,
                                  num_layers=num_enc_layers,
                                  cell_type=cell_type)
        emb_size=tgt_embedder.embedding_dim
        self.tgt_encoder = RNNEncoder(emb_size=emb_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  dropout=0.,
                                  num_layers=num_enc_layers,
                                  cell_type=cell_type)
        encoding_size = hidden_size if not bidirectional else hidden_size * 2
        self.normal_layer = NormalLayer(encoding_size*2, hidden_size, latent_size)

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        x_embed = self.src_embedder(x).detach()
        y_embed = self.tgt_embedder(y).detach()
        encoder_src_outputs, _ = self.src_encoder(x_embed, seq_len_x)
        encoder_tgt_outputs, _ = self.tgt_encoder(y_embed, seq_len_y)
        if self.max_pool:
            avg_encoder_src_output = encoder_src_outputs.max(dim=1)[0]
            avg_encoder_tgt_output = encoder_tgt_outputs.max(dim=1)[0]
        else:
            avg_encoder_src_output = (encoder_src_outputs * seq_mask_x.unsqueeze(-1).type_as(encoder_src_outputs)).sum(dim=1)
            avg_encoder_tgt_output = (encoder_tgt_outputs * seq_mask_y.unsqueeze(-1).type_as(encoder_tgt_outputs)).sum(dim=1)
        return self.normal_layer(torch.cat((avg_encoder_src_output,avg_encoder_tgt_output), dim=-1))

    def parameters(self, recurse=True):
        return chain(self.src_encoder.parameters(recurse=recurse), self.tgt_encoder.parameters(recurse=recurse), self.normal_layer.parameters(recurse=recurse))

    def named_parameters(self, prefix='', recurse=True):
        #TODO: am I sure that prefix needs to be changed?
        return chain(self.src_encoder.named_parameters(prefix='src', recurse=True), self.tgt_encoder.named_parameters(prefix='tgt', recurse=True)  ,  self.normal_layer.named_parameters(prefix='', recurse=True), )


class VAE(nn.Module):

    def __init__(self, emb_size, latent_size, hidden_size, bidirectional,num_layers,cell_type, language_model, max_pool,feed_z,pad_idx, dropout,language_model_tl):
        super().__init__()

        self.feed_z=feed_z
        self.latent_size = latent_size
        self.pad_idx = pad_idx

        self.language_model = language_model
        self.language_model_tl=language_model_tl

        self.dropout_layer = nn.Dropout(p=dropout)

        self.lm_init_layer = nn.Sequential(nn.Linear(latent_size, language_model.hidden_size),
                                           nn.Tanh())
        if self.language_model_tl is not None:
            self.lm_init_layer_tl = nn.Sequential(nn.Linear(latent_size, language_model_tl.hidden_size),
                                           nn.Tanh())
        else:
            self.lm_init_layer_tl =None

        if self.language_model_tl is not None:
            self.inf_network = BilingualInferenceNetwork(src_embedder=self.language_model.embedder,
                                                tgt_embedder=self.language_model_tl.embedder,
                                                hidden_size=hidden_size,
                                                latent_size=latent_size,
                                                bidirectional=bidirectional,
                                                num_enc_layers=num_layers,
                                                cell_type=cell_type,
                                                max_pool=max_pool)
        else:
            self.inf_network = InferenceNetwork(src_embedder=self.language_model.embedder,
                                                hidden_size=hidden_size,
                                                latent_size=latent_size,
                                                bidirectional=bidirectional,
                                                num_enc_layers=num_layers,
                                                cell_type=cell_type,
                                                max_pool=max_pool)
        self.pred_network=self.inf_network
        # This is done because the location and scale of the prior distribution are not considered
        # parameters, but are rather constant. Registering them as buffers still makes sure that
        # they will be moved to the appropriate device on which the model is run.
        self.register_buffer("prior_loc", torch.zeros([latent_size]))
        self.register_buffer("prior_scale", torch.ones([latent_size]))

    def inference_parameters(self):
        return self.inf_network.parameters()

    def generative_parameters(self):
        # TODO: separate the generative model into a GenerativeModel module
        #  within that module, have two modules, namely, LanguageModel and TranslationModel
        return self.lm_parameters()

    def lm_parameters(self):
        lm_tl_parameters=iter(())
        lm_tl_init_layer_parameters=iter(())
        if self.language_model_tl is not None:
            lm_tl_parameters=self.language_model_tl.parameters()
            lm_tl_init_layer_parameters=self.lm_init_layer_tl.parameters()
        return chain(self.language_model.parameters(), self.lm_init_layer.parameters(),lm_tl_parameters  ,lm_tl_init_layer_parameters )


    def approximate_posterior(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        """
        Returns an approximate posterior distribution q(z|x).
        """

        if self.language_model_tl is not None:
            return self.inf_network(x, seq_mask_x, seq_len_x,y, seq_mask_y, seq_len_y)
        else:
            return self.inf_network(x, seq_mask_x, seq_len_x)

    def approximate_posterior_prediction(self, x, seq_mask_x, seq_len_x):
        """
        Returns an approximate posterior distribution q(z|x).
        """

        return self.pred_network(x, seq_mask_x, seq_len_x)

    def prior(self):
        return torch.distributions.Normal(loc=self.prior_loc,
                                          scale=self.prior_scale)

    def src_embed(self, x):

        # We share the source embeddings with the language_model.
        x_embed = self.language_model.embedder(x)
        x_embed = self.dropout_layer(x_embed)
        return x_embed

    def tgt_embed(self,x):
        # We share the source embeddings with the language_model.
        x_embed = self.language_model_tl.embedder(x)
        x_embed = self.dropout_layer(x_embed)
        return x_embed


    def encode(self, x, seq_len_x, z):
        x_embed = self.src_embed(x)
        hidden = tile_rnn_hidden(self.encoder_init_layer(z), self.encoder.rnn)
        return self.encoder(x_embed, seq_len_x, hidden=hidden)


    def lm_generate(self, pre_output):
        return pre_output

    def init_lm(self, z):
        hidden = tile_rnn_hidden(self.lm_init_layer(z), self.language_model.rnn)
        return hidden

    def init_lm_tl(self, z):
        hidden = tile_rnn_hidden(self.lm_init_layer_tl(z), self.language_model_tl.rnn)
        return hidden

    def run_language_model(self, x, z):
        """
        Runs the language_model.

        :param x: unembedded source sentence
        :param z: a sample of the latent variable
        """
        hidden = tile_rnn_hidden(self.lm_init_layer(z), self.language_model.rnn)
        return self.language_model(x, hidden=hidden, z=z if self.feed_z else None)

    def run_language_model_tl(self, x, z):
        """
        Runs the language_model.

        :param x: unembedded source sentence
        :param z: a sample of the latent variable
        """
        hidden = tile_rnn_hidden(self.lm_init_layer_tl(z), self.language_model_tl.rnn)
        return self.language_model_tl(x, hidden=hidden, z=z if self.feed_z else None)

    def forward(self, x, seq_mask_x, seq_len_x, y, z):

        # Estimate the Categorical parameters for E[P(x|z)] using the given sample of the latent
        # variable.
        lm_logits = self.run_language_model(x, z)
        if self.language_model_tl is not None:
            lm_logits_tl=self.run_language_model_tl(y,z)
        else:
            lm_logits_tl=None

        return None, lm_logits, None,lm_logits_tl

    def compute_conditionals(self, x_in, seq_mask_x, seq_len_x, x_out, y_in, y_out, z):
        """
        :param x_in: [batch_size, max_length]
        :param seq_mask_x: [batch_size, max_length]
        :param seq_len_x: [batch_size]
        :param x_out: [batch_size, max_length]
        :param y_in: [batch_size, max_length]
        :param y_out: [batch_size, max_length]
        :param z: [batch_size, latent_size]
        :return: log p(x|z), log p(y|z,x)
        """
        raise NotImplementedError
        # Encode the source sentence and initialize the decoder hidden state.
        encoder_outputs, encoder_final = self.encode(x_in, seq_len_x, z)
        hidden = self.init_decoder(encoder_outputs, encoder_final, z)

        # Estimate the Categorical parameters for E[P(x|z)] using the given sample of the latent
        # variable.
        # [max_length, batch_size, vocab_size]
        lm_logits = self.run_language_model(x_in, z)

        # [batch_size, max_length, vocab_size]
        lm_logits = lm_logits.permute(0, 2, 1)

        # [batch_size]
        lm_loss = F.cross_entropy(lm_logits, x_out, ignore_index=self.pad_idx, reduction="none").sum(dim=1)

        return -lm_loss, 0

    def compute_lm_likelihood(self, x_in, seq_mask_x, seq_len_x, x_out, z):
        """
        :param x_in: [batch_size, max_length]
        :param seq_mask_x: [batch_size, max_length]
        :param seq_len_x: [batch_size]
        :param x_out: [batch_size, max_length]
        :param y_in: [batch_size, max_length]
        :param y_out: [batch_size, max_length]
        :param z: [batch_size, latent_size]
        :return: log p(x|z), log p(y|z,x)
        """
        raise NotImplementedError
        # Encode the source sentence and initialize the decoder hidden state.
        encoder_outputs, encoder_final = self.encode(x_in, seq_len_x, z)
        hidden = self.init_decoder(encoder_outputs, encoder_final, z)

        # Estimate the Categorical parameters for E[P(x|z)] using the given sample of the latent
        # variable.
        # [max_length, batch_size, vocab_size]
        lm_logits = self.run_language_model(x_in, z)
        # [batch_size, max_length, vocab_size]
        lm_logits = lm_logits.permute(0, 2, 1)

        lm_loss = F.cross_entropy(lm_logits, x_out, ignore_index=self.pad_idx, reduction="none").sum(dim=1)

        return -lm_loss


    #TODO: remove tm_logits? target_y?
    def loss(self, tm_logits, lm_logits, targets_y, targets_x, qz, free_nats=0.,
             KL_weight=1., reduction="mean", qz_prediction=None, lm_logits_tl=None):
        """
        Computes an estimate of the negative evidence lower bound for the single sample of the latent
        variable that was used to compute the categorical parameters, and the distributions qz
        that the sample comes from.

        :param tm_logits: translation model logits, the unnormalized translation probabilities [B, T_y, vocab_size]
        :param lm_logits: language model logits, the unnormalized language probabilities [B, T_x, vocab_size]
        :param targets_y: target labels target sentence [B, T_y]
        :param targets_x: target labels source sentence [B, T_x]
        :param qz: distribution that was used to sample the latent variable.
        :param free_nats: KL = min(free_nats, KL)
        :param KL_weight: weight to multiply the KL with, applied after free_nats
        :param reduction: what reduction to apply, none ([B]), mean ([]) or sum ([])
        """

        # Compute the language model categorical loss.
        lm_loss = self.language_model.loss(lm_logits, targets_x, reduction="none")

        if lm_logits_tl is not None:
            lm_loss_tl=self.language_model_tl.loss(lm_logits_tl,targets_y,reduction="none")
        else:
            lm_loss_tl=0

        # Compute the KL divergence between the distribution used to sample z, and the prior
        # distribution.
        pz = self.prior().expand(qz.mean.size())

        # The loss is the negative ELBO.
        lm_log_likelihood = -lm_loss - lm_loss_tl

        KL = torch.distributions.kl.kl_divergence(qz, pz)
        raw_KL = KL.sum(dim=1)
        KL = KL.sum(dim=1)

        if free_nats > 0:
            KL = torch.clamp(KL, min=free_nats)
        KL *= KL_weight
        elbo = lm_log_likelihood - KL
        loss = -elbo

        out_dict = {
            'lm_log_likelihood': lm_log_likelihood,
            'KL': KL,
            'raw_KL': raw_KL
        }

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