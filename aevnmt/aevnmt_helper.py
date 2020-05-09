from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.utils.data import DataLoader

from aevnmt.data import BucketingParallelDataLoader,BucketingTextDataLoader, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from aevnmt.data import create_batch, batch_to_sentences
from aevnmt.data.utils import create_noisy_batch
from aevnmt.components import DetachedEmbeddingLayer, RNNEncoder, beam_search, greedy_decode, sampling_decode
from aevnmt.models import AEVNMT
from aevnmt.models.generative import GenerativeLM, IndependentLM, CorrelatedBernoullisLM
from aevnmt.models.generative import CorrelatedCategoricalsLM, CorrelatedPoissonsLM
from aevnmt.models.generative import GenerativeTM, IndependentTM, CorrelatedBernoullisTM, CorrelatedCategoricalsTM
from aevnmt.models.generative import CorrelatedPoissonsTM, IBM1TM, AttentionBasedTM
from aevnmt.models.inference import InferenceModel, BasicInferenceModel, SwitchingInferenceModel
from aevnmt.models.inference import get_inference_encoder, combine_inference_encoders
from aevnmt.dist import get_named_params

from .train_utils import create_attention, create_encoder, create_decoder, attention_summary, compute_bleu


def _draw_translations(model, val_dl, vocab_src, vocab_tgt, device, hparams):
    with torch.no_grad():
        inputs = []
        references = []
        model_hypotheses = []
        for sentences_x, sentences_y in val_dl:
            hypothesis = translate(model, sentences_x, vocab_src, vocab_tgt, device, hparams, deterministic=False)

            # Keep track of inputs, references and model hypotheses.
            inputs += sentences_x.tolist()
            references += sentences_y.tolist()
            model_hypotheses += hypothesis.tolist()
    return inputs, references, model_hypotheses


def create_aux_language_models(vocab_src, src_embedder, hparams) -> Dict[str, GenerativeLM]:
    made_size=hparams.MADE_size
    if made_size == 0:
        made_size=hparams.hidden_size
    lms = dict()
    if hparams.bow_loss:
        lms['bow'] = IndependentLM(
            latent_size=hparams.latent_size,
            embedder=src_embedder,
            tied_embeddings=False,
            dropout=0.5)
    if hparams.MADE_loss:
        lms['made'] = CorrelatedBernoullisLM(
            vocab_size=src_embedder.num_embeddings,
            latent_size=hparams.latent_size,
            hidden_sizes=[made_size, made_size],  # TODO: generalise
            pad_idx=src_embedder.padding_idx,
            num_masks=10,  # TODO: generalise
            resample_mask_every=hparams.MADE_resample_mask_every,normalize_weight=hparams.MADE_normalize_weight_num_preds)  # TODO: generalise
    if hparams.count_MADE_loss:
        lms['count_made'] = CorrelatedPoissonsLM(
            vocab_size=src_embedder.num_embeddings,
            latent_size=hparams.latent_size,
            hidden_sizes=[made_size, made_size],  # TODO: generalise
            pad_idx=src_embedder.padding_idx,
            num_masks=10,  # TODO: generalise
            resample_mask_every=hparams.MADE_resample_mask_every,normalize_weight=hparams.MADE_normalize_weight_num_preds)  # TODO: generalise
    if hparams.shuffle_lm:  # TODO: implement shuffling
        lms['shuffled'] = CorrelatedCategoricalsLM(
            embedder=src_embedder,
            sos_idx=vocab_src[SOS_TOKEN],
            eos_idx=vocab_src[EOS_TOKEN],
            latent_size=hparams.latent_size,
            hidden_size=hparams.hidden_size,
            dropout=hparams.dropout,
            num_layers=hparams.num_dec_layers,
            cell_type=hparams.cell_type,
            tied_embeddings=hparams.tied_embeddings,
            feed_z=hparams.feed_z,
            gate_z=hparams.gate_z
        )
    return lms


def create_aux_translation_models(vocab_tgt,src_embedder, tgt_embedder, hparams) -> Dict[str, GenerativeTM]:
    made_size=hparams.MADE_size
    if made_size == 0:
        made_size=hparams.hidden_size
    tms = dict()
    if hparams.bow_loss_tl:
        tms['bow'] = IndependentTM(
            latent_size=hparams.latent_size,
            embedder=tgt_embedder,
            tied_embeddings=False,
            dropout=0.5)
    if hparams.MADE_loss_tl:
        tms['made'] = CorrelatedBernoullisTM(
            vocab_size=tgt_embedder.num_embeddings,
            latent_size=hparams.latent_size,
            hidden_sizes=[made_size, made_size],
            pad_idx=tgt_embedder.padding_idx,
            num_masks=10,  # TODO: generalise
            resample_mask_every=hparams.MADE_resample_mask_every,normalize_weight=hparams.MADE_normalize_weight_num_preds)  # TODO: generalise
    if hparams.count_MADE_loss_tl:
        tms['count_made'] = CorrelatedPoissonsTM(
            vocab_size=tgt_embedder.num_embeddings,
            latent_size=hparams.latent_size,
            hidden_sizes=[made_size, made_size],
            pad_idx=tgt_embedder.padding_idx,
            num_masks=10,  # TODO: generalise
            resample_mask_every=hparams.MADE_resample_mask_every,normalize_weight=hparams.MADE_normalize_weight_num_preds)  # TODO: generalise
    if hparams.shuffle_lm_tl:  # TODO: implement shuffling
        tms['shuffled'] = CorrelatedCategoricalsTM(
            embedder=tgt_embedder,
            sos_idx=vocab_tgt[SOS_TOKEN],
            eos_idx=vocab_tgt[EOS_TOKEN],
            latent_size=hparams.latent_size,
            hidden_size=hparams.hidden_size,
            dropout=hparams.dropout,
            num_layers=hparams.num_dec_layers,
            cell_type=hparams.cell_type,
            tied_embeddings=hparams.tied_embeddings,
            feed_z=hparams.feed_z,
            gate_z=hparams.gate_z
        )
    if hparams.ibm1_loss:
        tms['ibm1'] = IBM1TM(
            src_embed=src_embedder,
            latent_size=hparams.latent_size,
            hidden_size=hparams.hidden_size,
            src_vocab_size=src_embedder.num_embeddings,
            tgt_vocab_size=tgt_embedder.num_embeddings,
            pad_idx=tgt_embedder.padding_idx)
    return tms


def create_inference_model(src_embedder, tgt_embedder, hparams) -> InferenceModel:
    """Create an inference model and configure its encoder"""
    if not hparams.inf3:
        # Inference components
        inf_encoder = get_inference_encoder(
            encoder_style=hparams.inf_encoder_style,
            conditioning_context=hparams.inf_conditioning,
            embedder_x=src_embedder,
            embedder_y=tgt_embedder,
            hidden_size=hparams.hidden_size,
            rnn_bidirectional=hparams.bidirectional,
            rnn_num_layers=hparams.num_enc_layers,
            rnn_cell_type=hparams.cell_type,
            transformer_heads=hparams.transformer_heads,
            transformer_layers=hparams.num_enc_layers,
            nli_shared_size=hparams.emb_size,
            nli_max_distance=20,  # TODO: generalise
            dropout=hparams.dropout,
            composition="maxpool" if hparams.max_pooling_states else "avg")
        inf_model = BasicInferenceModel(
            family=hparams.posterior,
            latent_size=hparams.latent_size,
            hidden_size=hparams.hidden_size,
            encoder=inf_encoder)
    else:  # create 3 inference models and wrap them around a single container
        enc_styles = hparams.inf3.split(',')
        if len(enc_styles) != 3:
            raise ValueError("Specify exactly 3 comma-separated encoder styles, got '%s'" % hparams.inf3)
        encoder_x = get_inference_encoder(
            encoder_style=enc_styles[0],
            conditioning_context='x',
            embedder_x=src_embedder,
            embedder_y=tgt_embedder,
            hidden_size=hparams.hidden_size,
            rnn_bidirectional=hparams.bidirectional,
            rnn_num_layers=hparams.num_enc_layers,
            rnn_cell_type=hparams.cell_type,
            transformer_heads=hparams.transformer_heads,
            transformer_layers=hparams.num_enc_layers,
            nli_shared_size=hparams.emb_size,
            nli_max_distance=20,  # TODO: generalise
            dropout=hparams.dropout,
            composition="maxpool" if hparams.max_pooling_states else "avg")
        encoder_y = get_inference_encoder(
            encoder_style=enc_styles[1],
            conditioning_context='y',
            embedder_x=src_embedder,
            embedder_y=tgt_embedder,
            hidden_size=hparams.hidden_size,
            rnn_bidirectional=hparams.bidirectional,
            rnn_num_layers=hparams.num_enc_layers,
            rnn_cell_type=hparams.cell_type,
            transformer_heads=hparams.transformer_heads,
            transformer_layers=hparams.num_enc_layers,
            nli_shared_size=hparams.emb_size,
            nli_max_distance=20,  # TODO: generalise
            dropout=hparams.dropout,
            composition="maxpool" if hparams.max_pooling_states else "avg")
        if enc_styles[2] == 'comb':
            encoder_xy = combine_inference_encoders(encoder_x, encoder_y, hparams.inf3_comb_composition)
        elif enc_styles[2] == 'none':
            encoder_xy = None
        else:
            encoder_xy = get_inference_encoder(
                encoder_style=enc_styles[2],
                conditioning_context='xy',
                embedder_x=src_embedder,
                embedder_y=tgt_embedder,
                hidden_size=hparams.hidden_size,
                rnn_bidirectional=hparams.bidirectional,
                rnn_num_layers=hparams.num_enc_layers,
                rnn_cell_type=hparams.cell_type,
                transformer_heads=hparams.transformer_heads,
                transformer_layers=hparams.num_enc_layers,
                nli_shared_size=hparams.emb_size,
                nli_max_distance=20,  # TODO: generalise
                dropout=hparams.dropout,
                composition="maxpool" if hparams.max_pooling_states else "avg")
        inf_model = SwitchingInferenceModel(
            BasicInferenceModel(
                family=hparams.posterior,
                latent_size=hparams.latent_size,
                hidden_size=hparams.hidden_size,
                encoder=encoder_x),
            BasicInferenceModel(
                family=hparams.posterior,
                latent_size=hparams.latent_size,
                hidden_size=hparams.hidden_size,
                encoder=encoder_y),
            BasicInferenceModel(
                family=hparams.posterior,
                latent_size=hparams.latent_size,
                hidden_size=hparams.hidden_size,
                encoder=encoder_xy) if encoder_xy is not None else None,
            )
    return inf_model


def create_model(hparams, vocab_src, vocab_tgt):
    # Generative components
    src_embedder = torch.nn.Embedding(vocab_src.size(), hparams.emb_size, padding_idx=vocab_src[PAD_TOKEN])

    tgt_embedder=None
    if vocab_tgt is not None:
        tgt_embedder = torch.nn.Embedding(vocab_tgt.size(), hparams.emb_size, padding_idx=vocab_tgt[PAD_TOKEN])

    language_model = CorrelatedCategoricalsLM(
        embedder=src_embedder,
        sos_idx=vocab_src[SOS_TOKEN],
        eos_idx=vocab_src[EOS_TOKEN],
        latent_size=hparams.latent_size,
        hidden_size=hparams.hidden_size,
        dropout=hparams.dropout,
        num_layers=hparams.num_dec_layers,
        cell_type=hparams.cell_type,
        tied_embeddings=hparams.tied_embeddings,
        feed_z=hparams.feed_z,
        gate_z=hparams.gate_z
    )

    # Auxiliary generative components
    aux_lms = create_aux_language_models(vocab_src, src_embedder if not hparams.independent_embeddings_side else torch.nn.Embedding(vocab_src.size(), hparams.emb_size, padding_idx=vocab_src[PAD_TOKEN]), hparams)
    aux_tms = dict()
    translation_model=None

    if tgt_embedder is not None:
        aux_tms = create_aux_translation_models(vocab_tgt,src_embedder, tgt_embedder, hparams)
        encoder = create_encoder(hparams)
        attention = create_attention(hparams)
        decoder = create_decoder(attention, hparams)
        translation_model = AttentionBasedTM(
            src_embedder=src_embedder,
            tgt_embedder=tgt_embedder,
            tgt_sos_idx=vocab_tgt[SOS_TOKEN],
            tgt_eos_idx=vocab_tgt[EOS_TOKEN],
            encoder=encoder,
            decoder=decoder,
            latent_size=hparams.latent_size,
            dropout=hparams.dropout,
            feed_z=hparams.feed_z,
            tied_embeddings=hparams.tied_embeddings
        )

    # We want to give the inference model its own embedding layer

    if tgt_embedder is None:
        tgt_embedder_for_inf = None
    else:
        tgt_embedder_for_inf = DetachedEmbeddingLayer(tgt_embedder) if hparams.inf_share_embeddings else torch.nn.Embedding(
            tgt_embedder.num_embeddings, tgt_embedder.embedding_dim, padding_idx=tgt_embedder.padding_idx)

    inf_model = create_inference_model(
        DetachedEmbeddingLayer(src_embedder) if hparams.inf_share_embeddings else torch.nn.Embedding(
            src_embedder.num_embeddings, src_embedder.embedding_dim, padding_idx=src_embedder.padding_idx),
        tgt_embedder_for_inf,
        hparams)

    model = AEVNMT(
        latent_size=hparams.latent_size,
        src_embedder=src_embedder,
        tgt_embedder=tgt_embedder,
        language_model=language_model,
        translation_model=translation_model,
        inference_model=inf_model,
        dropout=hparams.dropout,
        feed_z=hparams.feed_z,
        tied_embeddings=hparams.tied_embeddings,
        prior_family=hparams.prior,
        prior_params=hparams.prior_params,
        aux_lms=aux_lms,
        aux_tms=aux_tms,
        mixture_likelihood=hparams.mixture_likelihood,
        mixture_likelihood_dir_prior=hparams.mixture_likelihood_dir_prior,
        mdr=hparams.mdr,
        lag_side=hparams.lag_side,
        lag_side_normtok=hparams.lag_side_normtok,
        lag_side_elbo=hparams.lag_side_elbo)
    return model

def train_step(model, x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in,
              x_shuf_in, x_shuf_out, seq_mask_x_shuf, seq_len_x_shuf,
              y_shuf_in, y_shuf_out, seq_mask_y_shuf, seq_len_y_shuf,
               hparams, step, summary_writer=None, synthetic_x=False,disable_main_loss=False, disable_side_losses=False):

    # Use q(z|x) for training to sample a z.
    qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x, y_in, seq_mask_y, seq_len_y)
    z = qz.rsample()

    # Compute the translation and language model logits.
    tm_likelihood, lm_likelihood, _, aux_lm_likelihoods, aux_tm_likelihoods = model(noisy_x_in, seq_mask_x, seq_len_x, noisy_y_in,
    x_shuf_in,seq_mask_x_shuf,seq_len_x_shuf,
    y_shuf_in,seq_mask_y_shuf,seq_len_y_shuf,
    z)

    # Do linear annealing of the KL over KL_annealing_steps if set.
    if hparams.KL_annealing_steps > 0:
        KL_weight = min(1., (1.0 / hparams.KL_annealing_steps) * step)
    else:
        KL_weight = 1.

    if synthetic_x:
        lm_likelihood=None
        aux_lm_likelihoods=dict()

    # Compute the loss.
    loss = model.loss(tm_likelihood, lm_likelihood, y_out, x_out,y_shuf_out,x_shuf_out, qz,
                      free_nats=hparams.KL_free_nats,
                      KL_weight=KL_weight,
                      reduction="mean",
                      aux_lm_likelihoods=aux_lm_likelihoods,
                      aux_tm_likelihoods=aux_tm_likelihoods,disable_main_loss=disable_main_loss, disable_side_losses=disable_side_losses)

    if summary_writer:
        summary_writer.add_histogram("posterior/z", z, step)
        for param_name, param_value in get_named_params(qz):
            summary_writer.add_histogram("posterior/%s" % param_name, param_value, step)
        pz = model.prior()
        # This part is perhaps not necessary for a simple prior (e.g. Gaussian),
        #  but it's useful for more complex priors (e.g. mixtures and NFs)
        prior_sample = pz.sample(torch.Size([z.size(0)]))
        summary_writer.add_histogram("prior/z", prior_sample, step)
        for param_name, param_value in get_named_params(pz):
            summary_writer.add_histogram("prior/%s" % param_name, param_value, step)

    return loss

def validate(model, val_data, vocab_src, vocab_tgt, device, hparams, step, title='xy', summary_writer=None,num_importance_samples=10):
    model.eval()

    # Create the validation dataloader. We can just bucket.
    val_dl = DataLoader(val_data, batch_size=hparams.batch_size,
                        shuffle=False, num_workers=4)
    if vocab_tgt is not None:
        val_dl = BucketingParallelDataLoader(val_dl)
    else:
        val_dl = BucketingTextDataLoader(val_dl)

    if hparams.log_KL_x_post_prior_length > 0:
        _compare_KL_posterior_prior(model, val_dl, vocab_src, vocab_tgt,hparams,step, summary_writer, device,target_len=hparams.log_KL_x_post_prior_length)
    val_ppl, val_KL, val_NLLs = _evaluate_perplexity(model, val_dl, vocab_src, vocab_tgt,hparams, device,num_importance_samples)
    val_NLL = val_NLLs['joint/main']
    if vocab_tgt is not None:
        val_bleu, inputs, refs, hyps = _evaluate_bleu(model, val_dl, vocab_src, vocab_tgt,
                                                  device, hparams)
    else:
        val_bleu=0.0

    #nll_str = ' '.join('-- validation NLL {} = {:.2f}'.format(comp_name, comp_value)  for comp_name, comp_value in sorted(val_NLLs.items()))
    nll_str = f""
    # - log P(x|z) for the various source LM decoders
    for comp_name, comp_nll in sorted(val_NLLs.items()):
        if comp_name.startswith('lm/'):
            nll_str += f" -- {comp_name} = {comp_nll:,.2f}"
    # - log P(y|z,x) for the various translation decoders
    for comp_name, comp_nll in sorted(val_NLLs.items()):
        if comp_name.startswith('tm/'):
            nll_str += f" -- {comp_name} = {comp_nll:,.2f}"

    if vocab_tgt is not None:
        random_idx = np.random.choice(len(inputs))
        random_source=inputs[random_idx]
        random_target=refs[random_idx]
        random_prediction=hyps[random_idx]
    else:
        random_source=random_target=random_prediction=""
    print(f"direction = {title}\n"
          f"validation perplexity = {val_ppl:,.2f}"
          f" -- BLEU = {val_bleu:.2f}"
          f" -- KL = {val_KL:.2f}"
          f" {nll_str}\n"
          f"- Source: {random_source}\n"
          f"- Target: {random_target}\n"
          f"- Prediction: {random_prediction}")

    if hparams.draw_translations > 0:
        random_idx = np.random.choice(len(inputs))
        dl = DataLoader([val_data[random_idx] for _ in range(hparams.draw_translations)], batch_size=hparams.batch_size, shuffle=False, num_workers=4)
        dl = BucketingParallelDataLoader(dl)
        i, r, hs = _draw_translations(model, dl, vocab_src, vocab_tgt, device, hparams)
        print("Posterior samples")
        print(f"- Input: {i[0]}")
        print(f"- Reference: {r[0]}")
        for h in hs:
            print(f"- Translation: {h}")

    # Write validation summaries.
    if summary_writer is not None:
        summary_writer.add_scalar(f"{title}/validation/BLEU", val_bleu, step)
        summary_writer.add_scalar(f"{title}/validation/perplexity", val_ppl, step)
        summary_writer.add_scalar(f"{title}/validation/KL", val_KL, step)
        for comp_name, comp_value in val_NLLs.items():
            summary_writer.add_scalar(f"{title}/validation/NLL/{comp_name}", comp_value, step)

        # Log the attention weights of the first validation sentence.
        if vocab_tgt is not None:
            with torch.no_grad():
                val_sentence_x, val_sentence_y = val_data[0]
                x_in, _, seq_mask_x, seq_len_x = create_batch([val_sentence_x], vocab_src, device)
                y_in, y_out, seq_mask_y, seq_len_y = create_batch([val_sentence_y], vocab_tgt, device)
                z = model.approximate_posterior(x_in, seq_mask_x, seq_len_x, y_in, seq_mask_y, seq_len_y).sample()
                _, _, state, _, _ = model(x_in, seq_mask_x, seq_len_x, y_in,None,None,None,
                None,None,None, z)
                att_weights = state['att_weights'].squeeze().cpu().numpy()
            src_labels = batch_to_sentences(x_in, vocab_src, no_filter=True)[0].split()
            tgt_labels = batch_to_sentences(y_out, vocab_tgt, no_filter=True)[0].split()
            attention_summary(src_labels, tgt_labels, att_weights, summary_writer,
                          f"{title}/validation/attention", step)

    side_NLL=sum( v for k,v in val_NLLs.items() if "/main" not in k  )
    return {'bleu': val_bleu, 'likelihood': -val_NLL, 'nll': val_NLL, 'ppl': val_ppl, 'side_NLL':side_NLL, 'side_likelihood': -side_NLL}


def re_sample(model, input_sentences, vocab_src,vocab_tgt, device, hparams, deterministic=True,z=None, use_prior=False,input_sentences_y=None, use_reverse_lm=False, force_first_token=None):
    model.eval()
    with torch.no_grad():
        if force_first_token is not None:
            force_first_token=vocab_src[force_first_token]
        x_in, _, seq_mask_x, seq_len_x = create_batch(input_sentences, vocab_src, device)
        if input_sentences_y is not None:
            y_in, _, seq_mask_y, seq_len_y = create_batch(input_sentences_y, vocab_tgt, device)

        if z is None:
            if model.prior_family == "mog" and use_prior:
                if deterministic:
                    raise NotImplementedError
                else:
                    z=model.prior().sample(torch.Size([ x_in.size(0) ]))
            else:
                if use_prior:
                    qz=model.prior().expand((x_in.size(0),))
                else:
                    if input_sentences_y is not None:
                        qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x,y_in, seq_mask_y, seq_len_y)
                    else:
                        qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x,None,None,None)
                z = qz.mean if deterministic else qz.sample()
        else:
            z=z.to(x_in.device)


        if not use_reverse_lm:
            language_model= model.language_model
        else:
            language_model= model.language_model_rev
        embed= model.src_embed
        vocab= vocab_src

        if hparams.sample_decoding:
            raw_hypothesis=language_model.sample(z)
        elif hparams.beam_width <= 1:
            raw_hypothesis=language_model.sample(z, greedy=True)
        else:
            raise NotImplementedError
        hypothesis = batch_to_sentences(raw_hypothesis, vocab)
        return hypothesis,z
################# OLD CODE ##############3333
        #if hparams.generate_homotopies:
        if False:
            NUM_STEPS=5
            z2=qz.sample()
            step=(z2-z)/NUM_STEPS

            z_list=[]
            for i in range(NUM_STEPS):
                z_list.append(z+i*step)
            z_list.append(z2)

            raw_hypothesis_l=[]
            for my_z in z_list:

                hidden = model.language_model.init(my_z)

                if hparams.sample_decoding:
                    raw_hypothesis_step = sampling_decode(language_model, embed,
                                                   model.language_model.generate, hidden,
                                                   None, None,z.size(0),
                                                   seq_mask_x, vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                                   vocab[PAD_TOKEN], hparams.max_decoding_length,hparams.sample_decoding_nucleus_p, my_z if hparams.feed_z else None, force_first_token=force_first_token)
                elif hparams.beam_width <= 1:
                    raw_hypothesis_step = greedy_decode(language_model, embed,
                                                   model.language_model.generate, hidden,
                                                   None, None,z.size(0),
                                                   seq_mask_x, vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                                   vocab[PAD_TOKEN], hparams.max_decoding_length,my_z if hparams.feed_z else None, force_first_token=force_first_token)
                else:
                    raw_hypothesis_step = beam_search(language_model, embed, model.language_model.generate,
                                                 vocab.size(), hidden, None,
                                                 None,z.size(0), seq_mask_x,
                                                 vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                                 vocab[PAD_TOKEN], hparams.beam_width,
                                                 hparams.length_penalty_factor,
                                                 hparams.max_decoding_length,hparams.n_best,my_z if hparams.feed_z else None, force_first_token=force_first_token)
                raw_hypothesis_l.append(raw_hypothesis_step)

            raw_hypothesis=torch.cat(raw_hypothesis_l,dim=1)


        else:
            hidden = model.language_model.init(z)

            if hparams.sample_decoding:
                raw_hypothesis = sampling_decode(language_model, embed,
                                               model.language_model.generate, hidden,
                                               None, None,z.size(0),
                                               seq_mask_x, vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                               vocab[PAD_TOKEN], hparams.max_decoding_length,hparams.sample_decoding_nucleus_p, z if hparams.feed_z else None, force_first_token=force_first_token)
            elif hparams.beam_width <= 1:
                raw_hypothesis = greedy_decode(language_model, embed,
                                               model.language_model.generate, hidden,
                                               None, None,z.size(0),
                                               seq_mask_x, vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                               vocab[PAD_TOKEN], hparams.max_decoding_length,z if hparams.feed_z else None, force_first_token=force_first_token)
            else:
                raw_hypothesis = beam_search(language_model, embed, model.language_model.generate,
                                             vocab.size(), hidden, None,
                                             None,z.size(0), seq_mask_x,
                                             vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                             vocab[PAD_TOKEN], hparams.beam_width,
                                             hparams.length_penalty_factor,
                                             hparams.max_decoding_length,hparams.n_best,z if hparams.feed_z else None, force_first_token=force_first_token)

    hypothesis = batch_to_sentences(raw_hypothesis, vocab)
    return hypothesis,z
########## END OLD CODE ##############


def translate(model, input_sentences, vocab_src, vocab_tgt, device, hparams, deterministic=True,z=None):
    # TODO: this code should be in the translation model class
    model.eval()
    with torch.no_grad():
        x_in, _, seq_mask_x, seq_len_x = create_batch(input_sentences, vocab_src, device)

        if z is None:
            # For translation we use the approximate posterior mean.
            qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x,
                    y=None, seq_mask_y=None, seq_len_y=None) # TODO: here we need a prediction net!
            z = qz.mean if deterministic else qz.sample()
        else:
            z=z.to(x_in.device)

        encoder_outputs, encoder_final = model.translation_model.encode(x_in, seq_len_x, z)
        hidden = model.translation_model.init_decoder(encoder_outputs, encoder_final, z)

        if hparams.sample_decoding:
            # TODO: we could use the new version below
            #raw_hypothesis = model.translation_model.sample(x_in, seq_mask_x, seq_len_x, z,
            #    max_len=hparams.max_decoding_length, greedy=False)
            raw_hypothesis = sampling_decode(
                model.translation_model.decoder,
                model.translation_model.tgt_embed,
                model.translation_model.generate, hidden,
                encoder_outputs, encoder_final,seq_mask_x.size(0),
                seq_mask_x, vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                vocab_tgt[PAD_TOKEN], hparams.max_decoding_length,
                z if hparams.feed_z else None)
        elif hparams.beam_width <= 1:
            # TODO: we could use the new version below
            #raw_hypothesis = model.translation_model.sample(x_in, seq_mask_x, seq_len_x, z,
            #    max_len=hparams.max_decoding_length, greedy=True)
            raw_hypothesis = greedy_decode(
                model.translation_model.decoder,
                model.translation_model.tgt_embed,
                model.translation_model.generate, hidden,
                encoder_outputs, encoder_final,seq_mask_x.size(0),
                seq_mask_x, vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                vocab_tgt[PAD_TOKEN], hparams.max_decoding_length,
                z if hparams.feed_z else None)
        else:
            raw_hypothesis = beam_search(
                model.translation_model.decoder,
                model.translation_model.tgt_embed,
                model.translation_model.generate,
                vocab_tgt.size(), hidden, encoder_outputs,
                encoder_final, seq_mask_x,
                vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                vocab_tgt[PAD_TOKEN], hparams.beam_width,
                hparams.length_penalty_factor,
                hparams.max_decoding_length,
                z if hparams.feed_z else None)

    hypothesis = batch_to_sentences(raw_hypothesis, vocab_tgt)
    return hypothesis,z

def _evaluate_bleu(model, val_dl, vocab_src, vocab_tgt, device, hparams):
    model.eval()
    with torch.no_grad():
        inputs = []
        references = []
        model_hypotheses = []
        for sentences_x, sentences_y in val_dl:
            hypothesis,zs = translate(model, sentences_x, vocab_src, vocab_tgt, device, hparams)

            # Keep track of inputs, references and model hypotheses.
            inputs += sentences_x.tolist()
            references += sentences_y.tolist()
            model_hypotheses += hypothesis.tolist()

    bleu = compute_bleu(model_hypotheses, references, subword_token=hparams.subword_token)
    return bleu, inputs, references, model_hypotheses

def _evaluate_perplexity(model, val_dl, vocab_src, vocab_tgt, hparams,device,num_importance_samples=10):
    model.eval()
    with torch.no_grad():
        num_predictions = 0
        num_sentences = 0
        log_marginal = defaultdict(float)
        total_KL = 0.
        n_samples = num_importance_samples
        for sentences_tuple in val_dl:
            if vocab_tgt is not None:
                sentences_x, sentences_y = sentences_tuple
            else:
                sentences_x = sentences_tuple
                sentences_y = None

            x_in, x_out, seq_mask_x, seq_len_x = create_batch(sentences_x, vocab_src, device)
            x_shuf_in, x_shuf_out, seq_mask_x_shuf, seq_len_x_shuf, noisy_x_shuf_in=create_noisy_batch(
                sentences_x, vocab_src, device,
                word_dropout=0.0,shuffle_toks=True,full_words_shuf=hparams.shuffle_lm_keep_bpe)


            if vocab_tgt is not None:
                y_in, y_out, seq_mask_y, seq_len_y = create_batch(sentences_y, vocab_tgt, device)
                y_shuf_in, y_shuf_out, seq_mask_y_shuf, seq_len_y_shuf, noisy_y_shuf_in=create_noisy_batch(
                    sentences_y, vocab_tgt, device,
                    word_dropout=0.0,shuffle_toks=True,full_words_shuf=hparams.shuffle_lm_keep_bpe)
            else:
                y_in=y_out=seq_mask_y=seq_len_y=None
                y_shuf_in=y_shuf_out=seq_mask_y_shuf=seq_len_y_shuf=noisy_y_shuf_in=None


            # Infer q(z|x) for this batch.
            qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x, y_in, seq_mask_y, seq_len_y)
            pz = model.prior()
            total_KL += torch.distributions.kl.kl_divergence(qz, pz).sum().item()

            # Take s importance samples from q(z|x):
            # log int{p(x, y, z) dz} ~= log sum_z{p(x, y, z) / q(z|x)} where z ~ q(z|x)
            batch_size = x_in.size(0)
            batch_log_marginals = defaultdict(lambda: torch.zeros(n_samples, batch_size))

            for s in range(n_samples):

                # z ~ q(z|x)
                z = qz.sample()

                # Compute the logits according to this sample of z.
                tm_likelihood, lm_likelihood, _, aux_lm_likelihoods, aux_tm_likelihoods = model(x_in, seq_mask_x, seq_len_x, y_in,
                noisy_x_shuf_in,seq_mask_x_shuf,seq_len_x_shuf,
                noisy_y_shuf_in,seq_mask_y_shuf,seq_len_y_shuf,
                z)

                # Compute log P(y|x, z_s)
                if  model.translation_model is not None:
                    log_tm_prob = model.translation_model.log_prob(tm_likelihood, y_out)
                else:
                    log_tm_prob = 0.0

                # Compute log P(x|z_s)
                log_lm_prob = model.language_model.log_prob(lm_likelihood, x_out)

                # Compute prior probability log P(z_s) and importance weight q(z_s|x)
                log_pz = pz.log_prob(z) # [B, latent_size] -> [B]
                log_qz = qz.log_prob(z)

                # Estimate the importance weighted estimate of (the log of) P(x, y)
                batch_log_marginals['joint/main'][s] = log_tm_prob + log_lm_prob + log_pz - log_qz
                batch_log_marginals['lm/main'][s] = log_lm_prob + log_pz - log_qz
                batch_log_marginals['tm/main'][s] = log_tm_prob + log_pz - log_qz

                for aux_comp, aux_px_z in aux_lm_likelihoods.items():
                    batch_log_marginals['lm/' + aux_comp][s] = model.log_likelihood_lm(aux_comp, aux_px_z,  x_shuf_out if aux_comp == "shuffled" else x_out) + log_pz - log_qz
                for aux_comp, aux_py_xz in aux_tm_likelihoods.items():
                    batch_log_marginals['tm/' + aux_comp][s] = model.log_likelihood_tm(aux_comp, aux_py_xz, y_shuf_out if aux_comp == "shuffled" else y_out) + log_pz - log_qz

            for comp_name, log_marginals in batch_log_marginals.items():
                # Average over all samples.
                batch_avg = torch.logsumexp(log_marginals, dim=0) - torch.log(torch.Tensor([n_samples]))
                log_marginal[comp_name] = log_marginal[comp_name] + batch_avg.sum().item()

            num_sentences += batch_size
            num_predictions += (seq_len_x.sum() + (seq_len_y.sum() if seq_len_y is not None else 0.0 ) ).item()

    val_NLL = -log_marginal['joint/main']
    val_perplexity = np.exp(val_NLL / num_predictions)

    NLLs = {comp_name: -value / num_sentences for comp_name, value in log_marginal.items()}

    return val_perplexity, total_KL/num_sentences, NLLs


def _compare_KL_posterior_prior(model, val_dl, vocab_src, vocab_tgt, hparams, step, summary_writer,device,target_len):
    model.eval()
    with torch.no_grad():
        num_predictions = 0
        num_sentences = 0
        log_marginal = defaultdict(float)
        total_KL = 0.
        n_samples = 10
        for sentences_tuple in val_dl:

            if vocab_tgt is not None:
                sentences_x, sentences_y = sentences_tuple
            else:
                sentences_x = sentences_tuple
                sentences_y = None

            x_in, x_out, seq_mask_x, seq_len_x = create_batch(sentences_x, vocab_src, device)
            x_shuf_in, x_shuf_out, seq_mask_x_shuf, seq_len_x_shuf, noisy_x_shuf_in=create_noisy_batch(
                sentences_x, vocab_src, device,
                word_dropout=0.0,shuffle_toks=True,full_words_shuf=hparams.shuffle_lm_keep_bpe)

            bsz=x_in.size(0)

            if vocab_tgt is not None:
                y_in, y_out, seq_mask_y, seq_len_y = create_batch(sentences_y, vocab_tgt, device)
                y_shuf_in, y_shuf_out, seq_mask_y_shuf, seq_len_y_shuf, noisy_y_shuf_in=create_noisy_batch(
                    sentences_y, vocab_tgt, device,
                    word_dropout=0.0,shuffle_toks=True,full_words_shuf=hparams.shuffle_lm_keep_bpe)
            else:
                y_in=y_out=seq_mask_y=seq_len_y=None
                y_shuf_in=y_shuf_out=seq_mask_y_shuf=seq_len_y_shuf=noisy_y_shuf_in=None


            # Infer q(z|x) for this batch.
            qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x, y_in, seq_mask_y, seq_len_y)
            pz = model.prior()

            # We are not doing importance sampling at the moment

            zposterior=qz.sample()
            zprior=torch.stack([pz.sample() for i in range(x_in.size(0))])

            # Compute the logits according to the posterior sample of z.
            tm_likelihood, lm_likelihood, _, aux_lm_likelihoods, aux_tm_likelihoods = model(x_in, seq_mask_x, seq_len_x, y_in,
            noisy_x_shuf_in,seq_mask_x_shuf,seq_len_x_shuf,
            noisy_y_shuf_in,seq_mask_y_shuf,seq_len_y_shuf,
            zposterior)

            # Compute the logits according to the prior sample of z.
            tm_likelihood_prior, lm_likelihood_prior, _, aux_lm_likelihoods_prior, aux_tm_likelihoods_prior = model(x_in, seq_mask_x, seq_len_x, y_in,
            noisy_x_shuf_in,seq_mask_x_shuf,seq_len_x_shuf,
            noisy_y_shuf_in,seq_mask_y_shuf,seq_len_y_shuf,
            zprior)


            if x_in.size(1) == target_len:
                for i in range(bsz):
                    #Discard sentences with padding
                    if torch.sum( y_out[i] == model.tgt_embedder.padding_idx ) == 0:
                        KLpostprior=torch.distributions.kl.kl_divergence(torch.distributions.Categorical(tm_likelihood.logits[i]),  torch.distributions.Categorical(tm_likelihood_prior.logits[i]))
                        KLpriorpost=torch.distributions.kl.kl_divergence( torch.distributions.Categorical(tm_likelihood_prior.logits[i]), torch.distributions.Categorical(tm_likelihood.logits[i]))
                        summary_writer.add_histogram('val-y/KLpriorpost',KLpriorpost,step)
                        summary_writer.add_histogram('val-y/KLpostprior',KLpostprior,step)


def product_of_gaussians(fwd_base: Normal, bwd_base: Normal) -> Normal:
    u1, s1, var1 = fwd_base.mean, fwd_base.stddev, fwd_base.variance
    u2, s2, var2 = bwd_base.mean, bwd_base.stddev, bwd_base.variance
    u = (var2 * u1 + var1 * u2) / (var1 + var2)
    s = 1. / (1. / var1 + 1. / var2)
    return Normal(u, s)


def mixture_of_gaussians(fwd_base: Normal, bwd_base: Normal) -> Normal:
    # [batch_size]
    selectors = torch.rand(fwd_base.mean.size(0), device=fwd_base.mean.device) >= 0.5
    # [batch_size, latent_size]
    selectors = selectors.unsqueeze(-1).repeat([1, fwd_base.mean.size(1)])
    u = torch.where(selectors, fwd_base.mean, bwd_base.mean)
    s = torch.where(selectors, fwd_base.stddev, bwd_base.stddev)
    return Normal(u, s)
