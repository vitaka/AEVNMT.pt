import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from aevnmt.components import RNNEncoder, tile_rnn_hidden, tile_rnn_hidden_for_decoder,TransformerEncoder
from aevnmt.dist import NormalLayer

import sparsedists.bernoulli

import dgm
from dgm.conditional import MADEConditioner
from dgm.likelihood import AutoregressiveLikelihood

from torch.distributions import Bernoulli

from itertools import chain

class InferenceNetwork(nn.Module):

    def __init__(self, src_embedder, hidden_size, latent_size, bidirectional, num_enc_layers, cell_type,max_pool,logvar, transformer_encoder=False,transformer_heads=8,transformer_hidden=2048):
        """
        :param src_embedder: uses this embedder, but detaches its output from the graph as to not compute
                             gradients for it.
        """
        super().__init__()
        self.max_pool=max_pool

        self.src_embedder = src_embedder
        self.transformer_encoder=transformer_encoder
        emb_size = src_embedder.embedding_dim
        if transformer_encoder:
            self.encoder= TransformerEncoder(input_size=emb_size,
                                         num_heads=transformer_heads,
                                         num_layers=num_enc_layers,
                                         dim_ff=transformer_hidden,
                                         dropout=0.)
            encoding_size=emb_size
        else:
            self.encoder = RNNEncoder(emb_size=emb_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  dropout=0.,
                                  num_layers=num_enc_layers,
                                  cell_type=cell_type)
            encoding_size = hidden_size if not bidirectional else hidden_size * 2
        self.normal_layer = NormalLayer(encoding_size, hidden_size, latent_size,logvar)

    def forward(self, x, seq_mask_x, seq_len_x,add_qz_scale=0.0):
        if self.transformer_encoder:
            #Add a special CLS token to the sentences? Problem: we do not know its embeddings
            x_embed = self.src_embedder(x).detach()
            encoder_outputs, _ = self.encoder(x_embed, seq_len_x) #(B, T, hidden_size)
            avg_encoder_output=encoder_outputs[:,0,:]
        else:
            x_embed = self.src_embedder(x).detach()
            encoder_outputs, _ = self.encoder(x_embed, seq_len_x) #(B, T, hidden_size)

            if self.max_pool:
                avg_encoder_output = encoder_outputs.max(dim=1)[0]
            else:
                avg_encoder_output = (encoder_outputs * seq_mask_x.unsqueeze(-1).type_as(encoder_outputs)).sum(dim=1)

        return self.normal_layer(avg_encoder_output,add_qz_scale)

    def parameters(self, recurse=True):
        return chain(self.encoder.parameters(recurse=recurse), self.normal_layer.parameters(recurse=recurse))

    def named_parameters(self, prefix='', recurse=True):
        return chain(self.encoder.named_parameters(prefix='', recurse=True), self.normal_layer.named_parameters(prefix='', recurse=True), )

class BilingualInferenceNetwork(nn.Module):

    def __init__(self, src_embedder, tgt_embedder, hidden_size, latent_size, bidirectional, num_enc_layers, cell_type, max_pool,logvar):
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
        self.normal_layer = NormalLayer(encoding_size*2, hidden_size, latent_size,logvar)

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y,add_qz_scale=0.0,disable_x=False,disable_y=False):
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
        if disable_x:
            avg_encoder_src_output=torch.zeros_like(avg_encoder_src_output)
        if disable_y:
            avg_encoder_tgt_output=torch.zeros_like(avg_encoder_tgt_output)
        return self.normal_layer(torch.cat((avg_encoder_src_output,avg_encoder_tgt_output), dim=-1),add_qz_scale)

    def parameters(self, recurse=True):
        return chain(self.src_encoder.parameters(recurse=recurse), self.tgt_encoder.parameters(recurse=recurse), self.normal_layer.parameters(recurse=recurse))

    def named_parameters(self, prefix='', recurse=True):
        #TODO: am I sure that prefix needs to be changed?
        return chain(self.src_encoder.named_parameters(prefix='src', recurse=True), self.tgt_encoder.named_parameters(prefix='tgt', recurse=True)  ,  self.normal_layer.named_parameters(prefix='', recurse=True), )


class VAE(nn.Module):

    def __init__(self, emb_size, latent_size, hidden_size, bidirectional,num_layers,cell_type, language_model, max_pool,feed_z,pad_idx, dropout,language_model_tl,language_model_rev,language_model_rev_tl,language_model_shuf,language_model_shuf_tl,masked_lm=None,masked_lm_mask_z_final=False,masked_lm_weight=1.0,masked_lm_proportion=0.15,masked_lm_bert=False,bow=False, disable_KL=False,logvar=False,bernoulli_bow=False,bernoulli_bow_norm_uniform=False,bernoulli_weight=True,MADE=False,transformer_inference_network=False):
        super().__init__()

        self.disable_KL=disable_KL
        self.logvar=logvar
        self.bernoulli_bow=bernoulli_bow
        self.bernoulli_bow_norm_uniform=bernoulli_bow_norm_uniform
        self.bernoulli_weight=bernoulli_weight

        self.feed_z=feed_z
        self.latent_size = latent_size
        self.pad_idx = pad_idx

        self.language_model = language_model
        self.language_model_tl=language_model_tl
        self.language_model_rev=language_model_rev
        self.language_model_rev_tl=language_model_rev_tl
        self.language_model_shuf=language_model_shuf
        self.language_model_shuf_tl=language_model_shuf_tl
        self.masked_lm=masked_lm
        self.masked_lm_mask_z_final=masked_lm_mask_z_final
        self.masked_lm_weight=masked_lm_weight
        self.masked_lm_proportion=masked_lm_proportion
        self.masked_lm_bert=masked_lm_bert

        self.dropout_layer = nn.Dropout(p=dropout)

        self.lm_init_layer = nn.Sequential(nn.Linear(latent_size, language_model.hidden_size),
                                           nn.Tanh())
        if self.language_model_tl is not None:
            self.lm_init_layer_tl = nn.Sequential(nn.Linear(latent_size, language_model_tl.hidden_size),
                                           nn.Tanh())
        else:
            self.lm_init_layer_tl =None

        if self.language_model_rev is not None:
            self.lm_init_layer_rev = nn.Sequential(nn.Linear(latent_size, language_model_rev.hidden_size),
                                           nn.Tanh())
        else:
            self.lm_init_layer_rev =None

        if self.language_model_rev_tl is not None:
            self.lm_init_layer_rev_tl = nn.Sequential(nn.Linear(latent_size, language_model_rev_tl.hidden_size),
                                           nn.Tanh())
        else:
            self.lm_init_layer_rev_tl =None

        if self.language_model_shuf is not None:
            self.lm_init_layer_shuf = nn.Sequential(nn.Linear(latent_size, language_model_shuf.hidden_size),
                                           nn.Tanh())
        else:
            self.lm_init_layer_shuf =None

        if self.language_model_shuf_tl is not None:
            self.lm_init_layer_shuf_tl = nn.Sequential(nn.Linear(latent_size, language_model_shuf_tl.hidden_size),
                                           nn.Tanh())
        else:
            self.lm_init_layer_shuf_tl =None

        if self.masked_lm is not None:
            self.masked_lm_linear_prediction=nn.Linear(self.masked_lm.input_size + ( self.latent_size if self.masked_lm_mask_z_final else 0),self.language_model.embedder.num_embeddings)
        else:
            self.masked_lm_linear_prediction=None

        if self.language_model_tl is not None:
            self.inf_network = BilingualInferenceNetwork(src_embedder=self.language_model.embedder,
                                                tgt_embedder=self.language_model_tl.embedder,
                                                hidden_size=hidden_size,
                                                latent_size=latent_size,
                                                bidirectional=bidirectional,
                                                num_enc_layers=num_layers,
                                                cell_type=cell_type,
                                                max_pool=max_pool,logvar=logvar)
        else:
            self.inf_network = InferenceNetwork(src_embedder=self.language_model.embedder,
                                                hidden_size=hidden_size,
                                                latent_size=latent_size,
                                                bidirectional=bidirectional,
                                                num_enc_layers=num_layers,
                                                cell_type=cell_type,
                                                max_pool=max_pool,logvar=logvar,transformer_encoder=transformer_inference_network)
        self.pred_network=self.inf_network

        self.bow_output_layer=None
        self.bow_output_layer_tl=None

        if bow:
            self.bow_output_layer = nn.Linear(latent_size,
                                              self.language_model.embedder.num_embeddings, bias=True)
            if self.language_model_tl is not None:
                self.bow_output_layer_tl = nn.Linear(latent_size,
                                                  self.language_model_tl.embedder.num_embeddings, bias=True)

        self.MADE=None
        self.MADE_tl=None
        if MADE:
            made = MADEConditioner(
                input_size= self.language_model.embedder.num_embeddings + self.latent_size,  # our only input to the MADE layer is the observation
                output_size= self.language_model.embedder.num_embeddings,  # number of parameters to predict
                context_size=self.latent_size,
                hidden_sizes=[8000, 8000], # TODO: is this OK?
                num_masks=2 #TODO: is that OK?
            )
            self.MADE = AutoregressiveLikelihood(
                event_size=self.language_model.embedder.num_embeddings,  # size of observation
                dist_type=Bernoulli,
                conditioner=made
                )
            if self.language_model_tl is not None:
                made_tl =MADEConditioner(
                    input_size= self.language_model_tl.embedder.num_embeddings  + self.latent_size,  # our only input to the MADE layer is the observation
                    output_size= self.language_model_tl.embedder.num_embeddings,  # number of parameters to predict
                    context_size=self.latent_size,
                    hidden_sizes=[8000, 8000], # TODO: is this OK?
                    num_masks=2 #TODO: is this OK?
                )
                self.MADE_tl= AutoregressiveLikelihood(
                    event_size=self.language_model_tl.embedder.num_embeddings,  # size of observation
                    dist_type=Bernoulli,
                    conditioner=made_tl
                    )

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
        return chain(self.lm_parameters(),self.bow_parameters())
        #return self.lm_parameters()

    def lm_parameters(self):
        lm_tl_parameters=iter(())
        lm_tl_init_layer_parameters=iter(())
        lm_rev_parameters=iter(())
        lm_rev_init_layer_parameters=iter(())
        lm_rev_tl_parameters=iter(())
        lm_rev_tl_init_layer_parameters=iter(())
        lm_shuf_parameters=iter(())
        lm_shuf_init_layer_parameters=iter(())
        lm_shuf_tl_parameters=iter(())
        lm_shuf_tl_init_layer_parameters=iter(())
        made_parameters=iter(())
        made_tl_parameters=iter(())

        masked_lm_parameters=iter(())
        if self.language_model_tl is not None:
            lm_tl_parameters=self.language_model_tl.parameters()
            lm_tl_init_layer_parameters=self.lm_init_layer_tl.parameters()
        if self.language_model_rev is not None:
            lm_rev_parameters=self.language_model_rev.parameters()
            lm_rev_init_layer_parameters=self.lm_init_layer_rev.parameters()
        if self.language_model_rev_tl is not None:
            lm_rev_tl_parameters=self.language_model_rev_tl.parameters()
            lm_rev_tl_init_layer_parameters=self.lm_init_layer_rev_tl.parameters()

        if self.language_model_shuf is not None:
            lm_shuf_parameters=self.language_model_shuf.parameters()
            lm_shuf_init_layer_parameters=self.lm_init_layer_shuf.parameters()
        if self.language_model_shuf_tl is not None:
            lm_shuf_tl_parameters=self.language_model_shuf_tl.parameters()
            lm_shuf_tl_init_layer_parameters=self.lm_init_layer_shuf_tl.parameters()

        if self.masked_lm is not None:
            masked_lm_parameters=chain(self.masked_lm.parameters(),self.masked_lm_linear_prediction.parameters())

        if self.MADE is not None:
            made_parameters=self.MADE.parameters()

        if self.MADE_tl is not None:
            made_tl_parameters=self.MADE_tl.parameters()

        return chain(self.language_model.parameters(), self.lm_init_layer.parameters(),lm_tl_parameters  ,lm_tl_init_layer_parameters , lm_rev_parameters, lm_rev_init_layer_parameters, lm_rev_tl_parameters, lm_rev_tl_init_layer_parameters,               lm_shuf_parameters, lm_shuf_init_layer_parameters, lm_shuf_tl_parameters, lm_shuf_tl_init_layer_parameters  , masked_lm_parameters , made_parameters, made_tl_parameters )

    def bow_parameters(self):
        return chain( iter(()) if self.bow_output_layer is None else self.bow_output_layer.parameters()   , iter(()) if self.bow_output_layer_tl is None else self.bow_output_layer_tl.parameters()  )


    def approximate_posterior(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y,add_qz_scale=0.0,disable_x=False, disable_y=False):
        """
        Returns an approximate posterior distribution q(z|x).
        """

        if self.language_model_tl is not None:
            return self.inf_network(x, seq_mask_x, seq_len_x,y, seq_mask_y, seq_len_y,add_qz_scale,disable_x,disable_y)
        else:
            return self.inf_network(x, seq_mask_x, seq_len_x,add_qz_scale)

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

    def run_language_model(self, x, z, reverse=False, shuffled=False):
        #TODO: shuf
        """
        Runs the language_model.

        :param x: unembedded source sentence
        :param z: a sample of the latent variable
        """
        lm=self.language_model
        init=self.lm_init_layer

        if reverse:
            lm=self.language_model_rev
            init=self.lm_init_layer_rev
        if shuffled:
            lm=self.language_model_shuf
            init=self.lm_init_layer_shuf

        hidden=tile_rnn_hidden(init(z), lm.rnn)
        return lm(x, hidden=hidden, z=z if self.feed_z else None)

    def run_language_model_tl(self, x, z, reverse=False, shuffled=False):
        """
        Runs the language_model.

        :param x: unembedded source sentence
        :param z: a sample of the latent variable
        """

        if reverse:
            lm=self.language_model_rev_tl
            init=self.lm_init_layer_rev_tl
        if shuffled:
            lm=self.language_model_shuf_tl
            init=self.lm_init_layer_shuf_tl

        hidden=tile_rnn_hidden(init(z), lm.rnn)
        return lm(x, hidden=hidden, z=z if self.feed_z else None)


    def forward(self, x, seq_mask_x, seq_len_x, y,x_rev,y_rev,x_shuf,y_shuf, z,disable_x=False, disable_y=False, x_mlm_masked=None, x_unshifted=None, y_unshifted=None):

        # Estimate the Categorical parameters for E[P(x|z)] using the given sample of the latent
        # variable.
        lm_logits = self.run_language_model(x, z) if not disable_x else None

        if self.language_model_tl is not None and not disable_y:
            lm_logits_tl=self.run_language_model_tl(y,z)
        else:
            lm_logits_tl=None

        lm_rev_logits=None
        lm_rev_logits_tl=None

        if self.language_model_rev is not None and not disable_x:
            lm_rev_logits=self.run_language_model(x_rev,z,reverse=True)

        if self.language_model_rev_tl is not None and not disable_y:
            lm_rev_logits_tl=self.run_language_model_tl(y_rev,z,reverse=True)

        lm_shuf_logits=None
        lm_shuf_logits_tl=None

        if self.language_model_shuf is not None and not disable_x:
            lm_shuf_logits=self.run_language_model(x_shuf,z,shuffled=True)

        if self.language_model_shuf_tl is not None and not disable_y:
            lm_shuf_logits_tl=self.run_language_model_tl(y_shuf,z,shuffled=True)

        #TODO: think about dropout too
        if self.bow_output_layer is not None and not disable_x:
            bow_logits=self.bow_output_layer(z)
        else:
            bow_logits=None

        if self.bow_output_layer_tl is not None and not disable_y:
            bow_logits_tl=self.bow_output_layer_tl(z)
        else:
            bow_logits_tl=None

        MADE_logits=None
        if self.MADE is not None:
            #Create binary inputs
            bsz=x.size(0)
            made_input=torch.zeros((bsz,self.MADE.event_size),device=x.device)
            for i in range(bsz):
                bow=torch.unique(x_unshifted[i] * seq_mask_x[i].type_as(x_unshifted[i]))
                made_input[i][bow]=1.0
            MADE_logits=self.MADE(z,made_input)

        MADE_logits_tl=None
        if self.MADE_tl is not None:
            #Create binary inputs
            bsz=x.size(0)
            made_input=torch.zeros((bsz,self.MADE_tl.event_size),device=x.device)
            for i in range(bsz):
                bow=torch.unique(y_unshifted[i] * seq_mask_y[i].type_as(y_unshifted[i]))
                made_input[i][bow]=1.0
            MADE_logits_tl=self.MADE_tl(z,made_input)

        masked_lm_logits=None
        if self.masked_lm is not None and x_mlm_masked is not None:
            masked_lm_result=self.masked_lm( self.language_model.embedder(x_mlm_masked), seq_len_x, z if not self.masked_lm_mask_z_final else None)[0]
            #masked_lm_result:[B, Tx, emb_size]
            #z: [B, latent_size]
            linear_input=masked_lm_result
            if self.masked_lm_mask_z_final:
                linear_input= torch.cat([linear_input, z.unsqueeze(1).repeat( 1 ,linear_input.size(1)  ,1) ], dim=-1)
            masked_lm_logits= self.masked_lm_linear_prediction ( linear_input )

        return None, lm_logits, None,lm_logits_tl,bow_logits, bow_logits_tl,lm_rev_logits,lm_rev_logits_tl,lm_shuf_logits,lm_shuf_logits_tl,masked_lm_logits, MADE_logits, MADE_logits_tl

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
    def loss(self, tm_logits, lm_logits, targets_y, targets_x, targets_y_rev,targets_x_rev,targets_y_shuf,targets_x_shuf, qz, free_nats=0.,free_nats_per_dimension=False,
             KL_weight=1., reduction="mean", qz_prediction=None, lm_logits_tl=None, bow_logits=None, bow_logits_tl=None,lm_rev_logits=None,lm_rev_logits_tl=None,lm_shuf_logits=None,lm_shuf_logits_tl=None, masked_lm_logits=None, masked_lm_mask=None,MADE_logits=None,MADE_logits_tl=None):
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
        :param bow_logits: [B,vocab_size]
        :param bow_logits_tl: [B,vocab_size_tl]
        """

        # Compute the language model categorical loss.
        lm_loss =0.0
        bow_loss=0.0
        MADE_loss=0.0
        if lm_logits is not None:
            lm_loss  = self.language_model.loss(lm_logits, targets_x, reduction="none")
            bow_loss=torch.zeros_like(lm_loss)
            MADE_loss=torch.zeros_like(lm_loss)

        if lm_logits_tl is not None:
            lm_loss_tl=self.language_model_tl.loss(lm_logits_tl,targets_y,reduction="none")
            bow_loss_tl=torch.zeros_like(lm_loss_tl)
            MADE_loss_tl=torch.zeros_like(lm_loss_tl)
        else:
            lm_loss_tl=0.0
            bow_loss_tl=0.0
            MADE_loss_tl=0.0

        lm_rev_loss=0.0
        lm_rev_loss_tl=0.0

        if lm_rev_logits is not None:
            lm_rev_loss=self.language_model_rev.loss(lm_rev_logits,targets_x_rev,reduction="none")

        if lm_rev_logits_tl is not None:
            lm_rev_loss_tl=self.language_model_rev_tl.loss(lm_rev_logits_tl,targets_y_rev,reduction="none")

        lm_shuf_loss=0.0
        lm_shuf_loss_tl=0.0

        if lm_shuf_logits is not None:
            lm_shuf_loss=self.language_model_shuf.loss(lm_shuf_logits,targets_x_shuf,reduction="none")

        if lm_shuf_logits_tl is not None:
            lm_shuf_loss_tl=self.language_model_shuf_tl.loss(lm_shuf_logits_tl,targets_y_shuf,reduction="none")

        masked_lm_loss=0.0
        if masked_lm_logits is not None:
            # Compute the loss for each batch element. Logits are of the form [B, T, vocab_size],
            # whereas the cross-entropy function wants a loss of the form [B, vocab_size, T].
            masked_lm_logits = masked_lm_logits.permute(0, 2, 1)
            masked_lm_loss = F.cross_entropy(masked_lm_logits, targets_x, ignore_index=self.pad_idx, reduction="none")*masked_lm_mask.float()
            masked_lm_loss = masked_lm_loss.sum(dim=1)

        num_bow_predictions=0
        if bow_logits is not None:
            if self.bernoulli_bow:
                bow_logprobs,bow_inverse_logprobs = sparsedists.bernoulli.bernoulli_log_probs_from_logit(bow_logits)
                bow_logprobs=-bow_logprobs
                bow_inverse_logprobs=-bow_inverse_logprobs
                bsz=bow_logits.size(0)

                for i in range(bsz):
                    bow=torch.unique(targets_x[i])
                    bow_mask=( bow != self.language_model.pad_idx)
                    bow=bow.masked_select(bow_mask)

                    vocab_mask=torch.ones_like(bow_logprobs[i])
                    vocab_mask[bow] = 0
                    vocab_mask[self.language_model.pad_idx]=0
                    inv_bow=vocab_mask.nonzero().squeeze()

                    bow_loss[i]=torch.sum( bow_logprobs[i][bow] ) + torch.sum( bow_inverse_logprobs[i][inv_bow] )
                    num_bow_predictions+=(len(bow) + len(inv_bow))

            else:
                bow_logprobs=-F.log_softmax(bow_logits,dim=-1)
                #bow_inverse_logprobs=-torch.log((1-torch.sigmoid(bow_logits)))
                bsz=bow_logits.size(0)
                for i in range(bsz):
                    bow=torch.unique(targets_x[i])
                    bow_mask=( bow != self.language_model.pad_idx)
                    bow=bow.masked_select(bow_mask)

                    #vocab_mask=torch.ones_like(bow_logprobs[i])
                    #vocab_mask[bow] = 0
                    #vocab_mask[self.language_model.pad_idx]=0
                    #inv_bow=vocab_mask.nonzero().squeeze()

                    bow_loss[i]=torch.sum( bow_logprobs[i][bow] ) #+ torch.sum( bow_inverse_logprobs[i][inv_bow] )
                    num_bow_predictions+=len(bow)


        if bow_logits_tl is not None:
            if self.bernoulli_bow:
                bow_logprobs_tl,bow_inverse_logprobs_tl = sparsedists.bernoulli.bernoulli_log_probs_from_logit(bow_logits_tl)
                bow_logprobs_tl=-bow_logprobs_tl
                bow_inverse_logprobs_tl=-bow_inverse_logprobs_tl
                bsz=bow_logits_tl.size(0)
                for i in range(bsz):
                    bow=torch.unique(targets_y[i])
                    bow_mask=( bow != self.language_model_tl.pad_idx)
                    bow=bow.masked_select(bow_mask)

                    vocab_mask=torch.ones_like(bow_logprobs_tl[i])
                    vocab_mask[bow] = 0
                    vocab_mask[self.language_model_tl.pad_idx]=0
                    inv_bow=vocab_mask.nonzero().squeeze()

                    bow_loss_tl[i]=torch.sum( bow_logprobs_tl[i][bow] ) + torch.sum( bow_inverse_logprobs_tl[i][inv_bow] )
                    num_bow_predictions+=(len(bow) + len(inv_bow))
            else:
                bow_logprobs_tl=-F.log_softmax(bow_logits_tl,dim=-1)
                #bow_inverse_logprobs_tl=-torch.log((1-torch.sigmoid(bow_logits_tl)))
                bsz=bow_logits_tl.size(0)
                for i in range(bsz):
                    bow=torch.unique(targets_y[i])
                    bow_mask=( bow != self.language_model_tl.pad_idx)
                    bow=bow.masked_select(bow_mask)

                    #vocab_mask=torch.ones_like(bow_logprobs_tl[i])
                    #vocab_mask[bow] = 0
                    #vocab_mask[self.language_model_tl.pad_idx]=0
                    #inv_bow=vocab_mask.nonzero().squeeze()

                    bow_loss_tl[i]=torch.sum( bow_logprobs_tl[i][bow] )# + torch.sum( bow_inverse_logprobs_tl[i][inv_bow] )
                    num_bow_predictions+=len(bow)

        if MADE_logits is not None:
            bsz=targets_x.size(0)
            made_ref=torch.zeros((bsz,self.MADE.event_size),device=targets_x.device)
            for i in range(bsz):
                bow=torch.unique(targets_x[i] )
                made_ref[i][bow]=1.0
            MADE_loss=-MADE_logits.log_prob(made_ref).sum(-1)
            if bow_logits is None:
                num_bow_predictions+=self.MADE.event_size

        if MADE_logits_tl is not None:
            bsz=targets_y.size(0)
            made_ref=torch.zeros((bsz,self.MADE_tl.event_size),device=targets_y.device)
            for i in range(bsz):
                bow=torch.unique(targets_y[i] )
                made_ref[i][bow]=1.0
            MADE_loss_tl= -MADE_logits_tl.log_prob(made_ref).sum(-1)
            if bow_logits is None:
                num_bow_predictions+=self.MADE_tl.event_size


        # Compute the KL divergence between the distribution used to sample z, and the prior
        # distribution.
        pz = self.prior().expand(qz.mean.size())

        #import pdb; pdb.set_trace()
        bow_weight=1
        if ((bow_logits is not None and self.bernoulli_bow) or  MADE_logits is not None ) and self.bernoulli_weight:
            #Ratio between number of LM predictions and number of bow predictions
            bow_weight= lm_logits.size(1)*1.0/(num_bow_predictions*1.0/lm_logits.size(0))
            if self.bernoulli_bow_norm_uniform:
                bow_weight= bow_weight*math.log(1/(num_bow_predictions*1.0/lm_logits.size(0)))/math.log(0.5)

        # The loss is the negative ELBO.
        lm_log_likelihood = -lm_loss - lm_loss_tl - lm_rev_loss - lm_rev_loss_tl - lm_shuf_loss - lm_shuf_loss_tl - masked_lm_loss*self.masked_lm_weight
        bow_log_likelihood = (- bow_loss - bow_loss_tl - MADE_loss -MADE_loss_tl )*bow_weight

        KL=0.0
        raw_KL=0.0

        qz_in=qz
        if not self.disable_KL:
            #print("qz.scale: {}".format(qz_in.scale))
            #print("pz.scale: {}".format(pz.scale))
            #var_ratio = (pz.scale / qz_in.scale).pow(2)
            #print("var_ratio: {}".format(var_ratio))
            KL = torch.distributions.kl.kl_divergence(qz_in, pz)
            raw_KL = KL.sum(dim=1)

            if free_nats > 0 and free_nats_per_dimension:
                KL = torch.clamp(KL, min=free_nats/KL.size(1))
            KL = KL.sum(dim=1)
            if free_nats > 0 and not free_nats_per_dimension:
                KL = torch.clamp(KL, min=free_nats)
            KL *= KL_weight

        elbo = lm_log_likelihood + bow_log_likelihood  - KL
        loss = -elbo

        out_dict = {
            'lm_log_likelihood': lm_log_likelihood,
            'bow_log_likelihood': bow_log_likelihood,
            'lr_lm_log_likelihood' : lm_loss,
            'rev_lm_log_likelihood' : lm_rev_loss,
            'shuf_lm_log_likelihood' : lm_shuf_loss,
            'masked_lm_log_likelihood' : masked_lm_loss,
            'bow_weight': bow_weight,
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
