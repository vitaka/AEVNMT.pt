import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

import sparsedists.bernoulli

from aevnmt.data import BucketingParallelDataLoader, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from aevnmt.data import create_batch, batch_to_sentences
from aevnmt.components import RNNEncoder, beam_search, greedy_decode, sampling_decode, TransformerEncoder
from aevnmt.models import AEVNMT, RNNLM, VAE
from .train_utils import create_attention, create_decoder, attention_summary, compute_bleu

from torch.utils.data import DataLoader


def _draw_translations(model, val_dl, vocab_src, vocab_tgt, device, hparams):
    with torch.no_grad():
        inputs = []
        references = []
        model_hypotheses = []
        for sentences_x, sentences_y in val_dl:
            input_sentences_y=None
            if hparams.disable_prediction_network and hparams.separate_prediction_network:
                input_sentences_y=sentences_y
            hypothesis_nbest,zs = translate(model, sentences_x, vocab_src, vocab_tgt, device, hparams, deterministic=False,input_sentences_y=input_sentences_y)
            hypothesis=[ t_nbest[0] for t_nbest in hypothesis_nbest]

            # Keep track of inputs, references and model hypotheses.
            inputs += sentences_x.tolist()
            references += sentences_y.tolist()
            model_hypotheses += hypothesis
    return inputs, references, model_hypotheses


def create_model(hparams, vocab_src, vocab_tgt):
    rnnlm = RNNLM(vocab_size=vocab_src.size(),
                  emb_size=hparams.emb_size,
                  hidden_size=hparams.hidden_size,
                  pad_idx=vocab_src[PAD_TOKEN],
                  dropout=hparams.dropout,
                  num_layers=hparams.num_dec_layers,
                  cell_type=hparams.cell_type,
                  tied_embeddings=hparams.tied_embeddings,
                  add_input_size= hparams.latent_size if hparams.feed_z else 0, gate_z=hparams.gate_z)

    rnnlm_tl=None
    if hparams.vae_tl_lm:
        rnnlm_tl = RNNLM(vocab_size=vocab_tgt.size(),
                      emb_size=hparams.emb_size,
                      hidden_size=hparams.hidden_size,
                      pad_idx=vocab_tgt[PAD_TOKEN],
                      dropout=hparams.dropout,
                      num_layers=hparams.num_dec_layers,
                      cell_type=hparams.cell_type,
                      tied_embeddings=hparams.tied_embeddings,
                      add_input_size= hparams.latent_size if hparams.feed_z else 0, gate_z=hparams.gate_z)

    rnnlm_rev=None
    rnnlm_tl_rev=None
    if hparams.reverse_lm:
        rnnlm_rev=RNNLM(vocab_size=vocab_src.size(),
                      emb_size=hparams.emb_size,
                      hidden_size=hparams.hidden_size,
                      pad_idx=vocab_src[PAD_TOKEN],
                      dropout=hparams.dropout,
                      num_layers=hparams.num_dec_layers,
                      cell_type=hparams.cell_type,
                      tied_embeddings=hparams.tied_embeddings,
                      add_input_size= hparams.latent_size if hparams.feed_z else 0, embedder=rnnlm.embedder if hparams.reverse_lm_shareemb else None, gate_z=hparams.gate_z)
        if  hparams.vae_tl_lm:
                rnnlm_tl_rev = RNNLM(vocab_size=vocab_tgt.size(),
                              emb_size=hparams.emb_size,
                              hidden_size=hparams.hidden_size,
                              pad_idx=vocab_tgt[PAD_TOKEN],
                              dropout=hparams.dropout,
                              num_layers=hparams.num_dec_layers,
                              cell_type=hparams.cell_type,
                              tied_embeddings=hparams.tied_embeddings,
                              add_input_size= hparams.latent_size if hparams.feed_z else 0, embedder=rnnlm_tl.embedder if hparams.reverse_lm_shareemb else None, gate_z=hparams.gate_z)

    rnnlm_shuf=None
    rnnlm_tl_shuf=None
    if hparams.shuffle_lm:
        rnnlm_shuf=RNNLM(vocab_size=vocab_src.size(),
                      emb_size=hparams.emb_size,
                      hidden_size=hparams.hidden_size,
                      pad_idx=vocab_src[PAD_TOKEN],
                      dropout=hparams.dropout,
                      num_layers=hparams.num_dec_layers,
                      cell_type=hparams.cell_type,
                      tied_embeddings=hparams.tied_embeddings,
                      add_input_size= hparams.latent_size if hparams.feed_z else 0, embedder=rnnlm.embedder if hparams.reverse_lm_shareemb else None, gate_z=hparams.gate_z)
        if  hparams.vae_tl_lm:
                rnnlm_tl_shuf = RNNLM(vocab_size=vocab_tgt.size(),
                              emb_size=hparams.emb_size,
                              hidden_size=hparams.hidden_size,
                              pad_idx=vocab_tgt[PAD_TOKEN],
                              dropout=hparams.dropout,
                              num_layers=hparams.num_dec_layers,
                              cell_type=hparams.cell_type,
                              tied_embeddings=hparams.tied_embeddings,
                              add_input_size= hparams.latent_size if hparams.feed_z else 0, embedder=rnnlm_tl.embedder if hparams.reverse_lm_shareemb else None, gate_z=hparams.gate_z)

    masked_lm=None
    if hparams.masked_lm:
        masked_lm=TransformerEncoder(input_size=hparams.emb_size,
                                     num_heads=hparams.transformer_heads,
                                     num_layers=hparams.num_enc_layers,
                                     dim_ff=hparams.transformer_hidden,
                                     dropout=hparams.dropout, z_size=hparams.latent_size if not hparams.masked_lm_mask_z_final else 0)

    model = VAE(   emb_size=hparams.emb_size,
                   latent_size=hparams.latent_size,
                   hidden_size=hparams.hidden_size,
                   bidirectional=hparams.bidirectional,
                   num_layers=hparams.num_enc_layers,
                   cell_type=hparams.cell_type,
                   language_model=rnnlm,
                   max_pool=hparams.max_pooling_states,
                   feed_z=hparams.feed_z,
                   pad_idx=vocab_tgt[PAD_TOKEN],
                   dropout=hparams.dropout,
                   language_model_tl=rnnlm_tl,
                   language_model_rev=rnnlm_rev,
                   language_model_rev_tl=rnnlm_tl_rev,
                   language_model_shuf=rnnlm_shuf,
                   language_model_shuf_tl=rnnlm_tl_shuf,
                   masked_lm=masked_lm,
                   masked_lm_mask_z_final=hparams.masked_lm_mask_z_final,
                   masked_lm_weight=1.0 if not hparams.masked_lm_weight_prop_prob else 1/hparams.masked_lm_mask_prob,
                   masked_lm_proportion=hparams.masked_lm_mask_prob,
                   bow=hparams.bow_loss,
                   disable_KL=hparams.disable_KL, logvar=hparams.logvar, bernoulli_bow=hparams.bow_loss_product_bernoulli)
    return model

def train_step(model, x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in,
               x_rev_in, x_rev_out, seq_mask_x_rev, seq_len_x_rev, noisy_x_rev_in, y_rev_in, y_rev_out, seq_mask_y_rev, seq_len_y_rev, noisy_y_rev_in,
               x_shuf_in, x_shuf_out, seq_mask_x_shuf, seq_len_x_shuf, noisy_x_shuf_in, y_shuf_in, y_shuf_out, seq_mask_y_shuf, seq_len_y_shuf, noisy_y_shuf_in,
               hparams, step,add_qz_scale=0.0, x_to_y=False,y_to_x=False):
    # Use q(z|x,y) for training to sample a z.
    qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x,y_in,seq_mask_y, seq_len_y, add_qz_scale, disable_x=y_to_x, disable_y=x_to_y)

    if model.disable_KL:
        z=qz.mean
    else:
        z = qz.rsample()

    #1= mask and predict
    #0=keep unchanged
    mlm_mask_positions=torch.rand(x_out.size(),device=seq_mask_x.device)
    mlm_mask_positions= (mlm_mask_positions <= model.masked_lm_proportion ) * seq_mask_x #boolean: True: mask, False: left intact
    inverse_mlm_mask_positions= ~ mlm_mask_positions # 1: left intact, 0: turn into mask

    # Compute the translation and language model logits.
    tm_logits, lm_logits, _, lm_logits_tl, bow_logits, bow_logits_tl,lm_rev_logits,lm_rev_logits_tl,lm_shuf_logits,lm_shuf_logits_tl, masked_lm_logits = model(noisy_x_in, seq_mask_x, seq_len_x, noisy_y_in, noisy_x_rev_in, noisy_y_rev_in, noisy_x_shuf_in, noisy_y_shuf_in, z,disable_x=x_to_y, disable_y=y_to_x,x_mlm_masked=x_out*inverse_mlm_mask_positions.long())

    # Do linear annealing of the KL over KL_annealing_steps if set.
    if hparams.KL_annealing_steps > 0:
        initial_KL=0
        KL_weight = min(1.,   ((1.0 - initial_KL ) / hparams.KL_annealing_steps) * step + initial_KL )
    else:
        KL_weight = 1.

    # Compute the loss.
    loss = model.loss(tm_logits, lm_logits, y_out, x_out,y_rev_out,x_rev_out,y_shuf_out,x_shuf_out, qz,
                      free_nats=hparams.KL_free_nats,free_nats_per_dimension=hparams.KL_free_nats_per_dimension,
                      KL_weight=KL_weight,
                      reduction="mean", qz_prediction=None,lm_logits_tl=lm_logits_tl,bow_logits=bow_logits, bow_logits_tl=bow_logits_tl,lm_rev_logits=lm_rev_logits,lm_rev_logits_tl=lm_rev_logits_tl,masked_lm_logits=masked_lm_logits,masked_lm_mask=mlm_mask_positions)
    loss["z"]=z
    return loss

def validate(model, val_data, vocab_src, vocab_tgt, device, hparams, step, title='xy', summary_writer=None):
    model.eval()

    # Create the validation dataloader. We can just bucket.
    val_dl = DataLoader(val_data, batch_size=hparams.batch_size,
                        shuffle=False, num_workers=4)
    val_dl = BucketingParallelDataLoader(val_dl,add_reverse=hparams.reverse_lm or hparams.shuffle_lm)

    val_ppl, val_NLL, val_KL, val_KL_pred, val_ppl_lm, val_NLL_lm, val_ppl_lm_rev, val_NLL_lm_rev,val_ppl_lm_shuf, val_NLL_lm_shuf, val_ppl_masked_lm, val_NLL_masked_lm = _evaluate_perplexity(model, val_dl, vocab_src, vocab_tgt, device)

    val_bleu_ref, inputs, refs, hyps = _evaluate_bleu(model, val_dl, vocab_src, vocab_tgt,
                                                  device, hparams)
    if hparams.paraphrasing_bleu:
        val_bleu_orig, inputs_orig, refs_orig, hyps_orig = _evaluate_bleu(model, val_dl, vocab_src, vocab_tgt, device, hparams, compare_with_original=True)
        val_bleu=val_bleu_ref-val_bleu_orig
        val_bleu_tl=0.0
    else:
        if hparams.vae_tl_lm:
            val_bleu_tl, inputs_tl, refs_tl, hyps_tl = _evaluate_bleu(model, val_dl, vocab_src, vocab_tgt, device, hparams, compare_with_original=True,generate_tl=True)
        else:
            val_bleu_tl=0
        val_bleu_orig=0
        val_bleu=val_bleu_ref+val_bleu_tl


    random_idx = np.random.choice(len(inputs))
    print(f"direction = {title}\n"
          f"validation perplexity = {val_ppl:,.2f}"
          f" -- validation NLL = {val_NLL:,.2f}"
          f" -- validation perplexity LM = {val_ppl_lm:,.2f}"
          f" -- validation NLL LM = {val_NLL_lm:,.2f}"
          f" -- validation perplexity LM rev = {val_ppl_lm_rev:,.2f}"
          f" -- validation NLL LM rev = {val_NLL_lm_rev:,.2f}"
          f" -- validation perplexity LM shuf = {val_ppl_lm_shuf:,.2f}"
          f" -- validation NLL LM shuf = {val_NLL_lm_shuf:,.2f}"
          f" -- validation perplexity masked LM = {val_ppl_masked_lm:,.2f}"
          f" -- validation NLL masked LM = {val_NLL_masked_lm:,.2f}"
          f" -- validation BLEU reference= {val_bleu_ref:.2f}"
          f" -- validation BLEU original = {val_bleu_orig:.2f}"
          f" -- validation BLEU TL = {val_bleu_tl:.2f}"
          f" -- validation BLEU = {val_bleu:.2f}"
          f" -- validation KL = {val_KL:.2f}\n"
          f"- Source: {inputs[random_idx]}\n"
          f"- Target: {refs[random_idx]}\n"
          f"- Prediction: {hyps[random_idx]}")
    if hparams.vae_tl_lm:
          print( f"- Source: {inputs_tl[random_idx]}\n"
          f"- Target: {refs_tl[random_idx]}\n"
          f"- Prediction: {hyps_tl[random_idx]}")

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
        summary_writer.add_scalar(f"{title}/validation/NLL", val_NLL, step)
        summary_writer.add_scalar(f"{title}/validation/BLEU", val_bleu, step)
        summary_writer.add_scalar(f"{title}/validation/perplexity", val_ppl, step)
        summary_writer.add_scalar(f"{title}/validation/NLL_LM", val_NLL_lm, step)
        summary_writer.add_scalar(f"{title}/validation/NLL_rev_LM", val_NLL_lm_rev, step)
        summary_writer.add_scalar(f"{title}/validation/NLL_shuf_LM", val_NLL_lm_shuf, step)
        summary_writer.add_scalar(f"{title}/validation/NLL_masked_LM", val_NLL_masked_lm, step)

    return {'bleu': val_bleu, 'likelihood': -val_NLL, 'nll': val_NLL, 'ppl': val_ppl, 'main_likelihood': -val_NLL_lm}


def re_sample(model, input_sentences, vocab_src,vocab_tgt, device, hparams, deterministic=True,z=None, use_prior=False,input_sentences_y=None, use_tl_lm=False, use_reverse_lm=False):
    model.eval()
    with torch.no_grad():
        x_in, _, seq_mask_x, seq_len_x = create_batch(input_sentences, vocab_src, device)
        if input_sentences_y is not None:
            y_in, _, seq_mask_y, seq_len_y = create_batch(input_sentences_y, vocab_tgt, device)

        if z is None:
            if input_sentences_y is not None:
                qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x,y_in, seq_mask_y, seq_len_y)
            else:
                qz = model.approximate_posterior_prediction(x_in, seq_mask_x, seq_len_x)
            if use_prior:
                #TODO:We are computing qz and it is not needed
                qz=model.prior().expand(qz.mean.size())
            z = qz.mean if deterministic else qz.sample()
        else:
            z=z.to(x_in.device)


        if not use_reverse_lm:
            language_model=model.language_model_tl if use_tl_lm else model.language_model
        else:
            language_model=model.language_model_rev_tl if use_tl_lm else model.language_model_rev
        embed=model.tgt_embed if use_tl_lm else model.src_embed
        vocab=vocab_tgt if use_tl_lm else vocab_src

        if hparams.generate_homotopies:
            NUM_STEPS=5
            z2=qz.sample()
            step=(z2-z)/NUM_STEPS

            z_list=[]
            for i in range(NUM_STEPS):
                z_list.append(z+i*step)
            z_list.append(z2)

            raw_hypothesis_l=[]
            for my_z in z_list:
                if use_tl_lm:
                    hidden=model.init_lm_tl(my_z)
                else:
                    hidden = model.init_lm(my_z)

                if hparams.sample_decoding:
                    raw_hypothesis_step = sampling_decode(language_model, embed,
                                                   model.lm_generate, hidden,
                                                   None, None,
                                                   seq_mask_x, vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                                   vocab[PAD_TOKEN], hparams.max_decoding_length,hparams.sample_decoding_nucleus_p, my_z if hparams.feed_z else None)
                elif hparams.beam_width <= 1:
                    raw_hypothesis_step = greedy_decode(language_model, embed,
                                                   model.lm_generate, hidden,
                                                   None, None,
                                                   seq_mask_x, vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                                   vocab[PAD_TOKEN], hparams.max_decoding_length,my_z if hparams.feed_z else None)
                else:
                    raw_hypothesis_step = beam_search(language_model, embed, model.lm_generate,
                                                 vocab.size(), hidden, None,
                                                 None, seq_mask_x,
                                                 vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                                 vocab[PAD_TOKEN], hparams.beam_width,
                                                 hparams.length_penalty_factor,
                                                 hparams.max_decoding_length,hparams.n_best,my_z if hparams.feed_z else None)
                raw_hypothesis_l.append(raw_hypothesis_step)

            raw_hypothesis=torch.cat(raw_hypothesis_l,dim=1)



        else:
            if use_tl_lm:
                hidden=model.init_lm_tl(z)
            else:
                hidden = model.init_lm(z)

            if hparams.sample_decoding:
                raw_hypothesis = sampling_decode(language_model, embed,
                                               model.lm_generate, hidden,
                                               None, None,
                                               seq_mask_x, vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                               vocab[PAD_TOKEN], hparams.max_decoding_length,hparams.sample_decoding_nucleus_p, z if hparams.feed_z else None)
            elif hparams.beam_width <= 1:
                raw_hypothesis = greedy_decode(language_model, embed,
                                               model.lm_generate, hidden,
                                               None, None,
                                               seq_mask_x, vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                               vocab[PAD_TOKEN], hparams.max_decoding_length,z if hparams.feed_z else None)
            else:
                raw_hypothesis = beam_search(language_model, embed, model.lm_generate,
                                             vocab.size(), hidden, None,
                                             None, seq_mask_x,
                                             vocab[SOS_TOKEN], vocab[EOS_TOKEN],
                                             vocab[PAD_TOKEN], hparams.beam_width,
                                             hparams.length_penalty_factor,
                                             hparams.max_decoding_length,hparams.n_best,z if hparams.feed_z else None)

    #hypothesis_l: size= nbest
    hypothesis_l=[]
    for n in range(raw_hypothesis.size(1)):
        hypothesis_l.append(batch_to_sentences(raw_hypothesis[:,n,:], vocab))

    return np.array(hypothesis_l).transpose(1, 0),z

def translate(model, input_sentences, vocab_src, vocab_tgt, device, hparams, deterministic=True,z=None,use_prior=False,input_sentences_y=None):
    model.eval()
    with torch.no_grad():
        x_in, _, seq_mask_x, seq_len_x = create_batch(input_sentences, vocab_src, device)
        if input_sentences_y is not None:
            y_in, _, seq_mask_y, seq_len_y = create_batch(input_sentences_y, vocab_tgt, device)

        if z is None:
            # For translation we use the approximate posterior mean.
            if input_sentences_y is not None:
                qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x,y_in, seq_mask_y, seq_len_y)
            else:
                qz = model.approximate_posterior_prediction(x_in, seq_mask_x, seq_len_x)
            if use_prior:
                #TODO:We are computing qz and it is not need
                qz=model.prior().expand(qz.mean.size())
            z = qz.mean if deterministic else qz.sample()
        else:
            #Ensure it is loaded in GPU
            z=z.to(x_in.device)

        encoder_outputs, encoder_final = model.encode(x_in, seq_len_x, z)
        hidden = model.init_decoder(encoder_outputs, encoder_final, z)

        if hparams.sample_decoding:
            raw_hypothesis = sampling_decode(model.decoder, model.tgt_embed,
                                           model.generate, hidden,
                                           encoder_outputs, encoder_final,
                                           seq_mask_x, vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                                           vocab_tgt[PAD_TOKEN], hparams.max_decoding_length,hparams.sample_decoding_nucleus_p)
        elif hparams.beam_width <= 1:
            raw_hypothesis = greedy_decode(model.decoder, model.tgt_embed,
                                           model.generate, hidden,
                                           encoder_outputs, encoder_final,
                                           seq_mask_x, vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                                           vocab_tgt[PAD_TOKEN], hparams.max_decoding_length)
        else:
            raw_hypothesis = beam_search(model.decoder, model.tgt_embed, model.generate,
                                         vocab_tgt.size(), hidden, encoder_outputs,
                                         encoder_final, seq_mask_x,
                                         vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                                         vocab_tgt[PAD_TOKEN], hparams.beam_width,
                                         hparams.length_penalty_factor,
                                         hparams.max_decoding_length,hparams.n_best)

    hypothesis_l=[]
    for n in range(raw_hypothesis.size(1)):
        hypothesis_l.append(batch_to_sentences(raw_hypothesis[:,n,:], vocab_tgt))

    return np.array(hypothesis_l).transpose(1, 0),z

def _evaluate_bleu(model, val_dl, vocab_src, vocab_tgt, device, hparams, compare_with_original=False, generate_tl=False):
    model.eval()
    with torch.no_grad():
        inputs = []
        references = []
        model_hypotheses = []
        for sentences_tuple in val_dl:
            sentences_x=sentences_tuple[0]
            sentences_y=sentences_tuple[1]
            if getattr(model,'language_model_tl',None) is not None:
                input_sentences_y=sentences_y
            else:
                input_sentences_y=None
            hypothesis_nbest,zs = re_sample(model, sentences_x, vocab_src, vocab_tgt, device, hparams,input_sentences_y=input_sentences_y,use_tl_lm=generate_tl)
            hypothesis=[ t_nbest[0] for t_nbest in hypothesis_nbest]

            # Keep track of inputs, references and model hypotheses.
            inputs += sentences_x.tolist()
            if compare_with_original or generate_tl:
                references += sentences_y.tolist()
            else:
                references += sentences_x.tolist()
            model_hypotheses += hypothesis

    bleu = compute_bleu(model_hypotheses, references, subword_token=hparams.subword_token)
    return bleu, inputs, references, model_hypotheses

def _evaluate_perplexity(model, val_dl, vocab_src, vocab_tgt, device):
    model.eval()
    with torch.no_grad():
        num_predictions = 0
        num_predictions_lm = 0
        num_predictions_masked_lm = 0
        num_sentences = 0
        log_marginal = 0.
        log_marginal_lm = 0.0
        log_marginal_lm_rev = 0.0
        log_marginal_lm_shuf = 0.0
        log_marginal_masked_lm = 0.0

        total_KL = 0.
        total_KL_prediction = 0.0
        n_samples = 10
        if model.disable_KL:
            n_samples=1
        for sentences_tuple in val_dl:
            if model.language_model_rev is not None or model.language_model_shuf is not None:
                sentences_x, sentences_y, sentences_x_rev, sentences_y_rev, sentences_x_shuf, sentences_y_shuf = sentences_tuple
            else:
                sentences_x, sentences_y = sentences_tuple

            x_in, x_out, seq_mask_x, seq_len_x = create_batch(sentences_x, vocab_src, device)
            y_in, y_out, seq_mask_y, seq_len_y = create_batch(sentences_y, vocab_tgt, device)

            if model.language_model_rev is not None:
                x_rev_in, x_rev_out, seq_mask_x_rev, seq_len_x_rev = create_batch(sentences_x_rev, vocab_src, device)
                y_rev_in, y_rev_out, seq_mask_y_rev, seq_len_y_rev = create_batch(sentences_y_rev, vocab_tgt, device)
            else:
                x_rev_in= x_rev_out= seq_mask_x_rev= seq_len_x_rev=None
                y_rev_in= y_rev_out= seq_mask_y_rev= seq_len_y_rev=None

            if model.language_model_shuf is not None:
                x_shuf_in, x_shuf_out, seq_mask_x_shuf, seq_len_x_shuf = create_batch(sentences_x_shuf, vocab_src, device)
                y_shuf_in, y_shuf_out, seq_mask_y_shuf, seq_len_y_shuf = create_batch(sentences_y_shuf, vocab_tgt, device)
            else:
                x_shuf_in= x_shuf_out= seq_mask_x_shuf= seq_len_x_shuf=None
                y_shuf_in= y_shuf_out= seq_mask_y_shuf= seq_len_y_shuf=None


            # Infer q(z|x) for this batch.
            qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x, y_in, seq_mask_y, seq_len_y)
            pz = model.prior().expand(qz.mean.size())
            total_KL += torch.distributions.kl.kl_divergence(qz, pz).sum().item()

            # Take s importance samples from q(z|x):
            # log int{p(x, y, z) dz} ~= log sum_z{p(x, y, z) / q(z|x)} where z ~ q(z|x)
            batch_size = x_in.size(0)
            batch_log_marginals = torch.zeros(n_samples, batch_size)
            batch_log_marginals_lm = torch.zeros(n_samples, batch_size)
            batch_log_marginals_lm_rev = torch.zeros(n_samples, batch_size)
            batch_log_marginals_lm_shuf = torch.zeros(n_samples, batch_size)
            batch_log_marginals_masked_lm = torch.zeros(n_samples, batch_size)

            #Compute bow for each sentence
            bow_indexes=[]
            bow_indexes_inv=[]
            if model.bow_output_layer is not None:
                for i in range(batch_size):
                    bow=torch.unique(x_out[i] * seq_mask_x[i].type_as(x_out[i]))
                    bow_mask=( bow != 0)
                    bow=bow.masked_select(bow_mask)
                    bow_indexes.append(bow)

                    vocab_mask=torch.ones(model.bow_output_layer.out_features,device=x_out.device)
                    vocab_mask[bow] = 0
                    vocab_mask[model.language_model.pad_idx]=0
                    inv_bow=vocab_mask.nonzero().squeeze()
                    bow_indexes_inv.append(inv_bow)

            bow_indexes_tl=[]
            bow_indexes_inv_tl=[]
            if model.bow_output_layer_tl is not None:
                for i in range(batch_size):
                    bow=torch.unique(y_out[i] * seq_mask_y[i].type_as(y_out[i]),dim=-1)
                    bow_mask=( bow != 0)
                    bow=bow.masked_select(bow_mask)
                    bow_indexes_tl.append(bow)

                    vocab_mask=torch.ones(model.bow_output_layer_tl.out_features,device=x_out.device)
                    vocab_mask[bow] = 0
                    vocab_mask[model.language_model_tl.pad_idx]=0
                    inv_bow=vocab_mask.nonzero().squeeze()
                    bow_indexes_inv_tl.append(inv_bow)


            for s in range(n_samples):

                # z ~ q(z|x)
                if model.disable_KL:
                    z=qz.mean
                else:
                    z = qz.sample()

                if model.masked_lm is not None:
                    mlm_mask_positions=torch.rand(x_out.size(),device=seq_mask_x.device)
                    mlm_mask_positions= (mlm_mask_positions <= model.masked_lm_proportion ) * seq_mask_x #boolean: True: mask, False: left intact
                    inverse_mlm_mask_positions= ~ mlm_mask_positions # 1: left intact, 0: turn into mask

                # Compute the logits according to this sample of z.
                _, lm_logits, _ , lm_logits_tl,bow_logits, bow_logits_tl,lm_rev_logits,lm_rev_logits_tl,lm_shuf_logits,lm_shuf_logits_tl, masked_lm_logits  = model(x_in, seq_mask_x, seq_len_x, y_in,x_rev_in,y_rev_in,x_shuf_in,y_shuf_in, z,x_mlm_masked=x_out*inverse_mlm_mask_positions.long())

                # Compute log P(x|z_s)
                log_lm_prob = F.log_softmax(lm_logits, dim=-1)
                log_lm_prob = torch.gather(log_lm_prob, 2, x_out.unsqueeze(-1)).squeeze()
                log_lm_prob = (seq_mask_x.type_as(log_lm_prob) * log_lm_prob).sum(dim=1)

                log_lm_prob_tl=0.0
                if lm_logits_tl is not None:
                    log_lm_prob_tl = F.log_softmax(lm_logits_tl, dim=-1)
                    log_lm_prob_tl = torch.gather(log_lm_prob_tl, 2, y_out.unsqueeze(-1)).squeeze()
                    log_lm_prob_tl = (seq_mask_y.type_as(log_lm_prob_tl) * log_lm_prob_tl).sum(dim=1)

                log_lm_prob_rev=0.0
                if lm_rev_logits is not None:
                    log_lm_prob_rev = F.log_softmax(lm_rev_logits, dim=-1)
                    log_lm_prob_rev = torch.gather(log_lm_prob_rev, 2, x_rev_out.unsqueeze(-1)).squeeze()
                    log_lm_prob_rev = (seq_mask_x_rev.type_as(log_lm_prob_rev) * log_lm_prob_rev).sum(dim=1)

                log_lm_prob_rev_tl=0.0
                if lm_rev_logits_tl is not None:
                    log_lm_prob_rev_tl = F.log_softmax(lm_rev_logits_tl, dim=-1)
                    log_lm_prob_rev_tl = torch.gather(log_lm_prob_rev_tl, 2, y_rev_out.unsqueeze(-1)).squeeze()
                    log_lm_prob_rev_tl = (seq_mask_y_rev.type_as(log_lm_prob_rev_tl) * log_lm_prob_rev_tl).sum(dim=1)

                log_lm_prob_shuf=0.0
                if lm_shuf_logits is not None:
                    log_lm_prob_shuf = F.log_softmax(lm_shuf_logits, dim=-1)
                    log_lm_prob_shuf = torch.gather(log_lm_prob_shuf, 2, x_shuf_out.unsqueeze(-1)).squeeze()
                    log_lm_prob_shuf = (seq_mask_x_shuf.type_as(log_lm_prob_shuf) * log_lm_prob_shuf).sum(dim=1)

                log_lm_prob_shuf_tl=0.0
                if lm_shuf_logits_tl is not None:
                    log_lm_prob_shuf_tl = F.log_softmax(lm_shuf_logits_tl, dim=-1)
                    log_lm_prob_shuf_tl = torch.gather(log_lm_prob_shuf_tl, 2, y_shuf_out.unsqueeze(-1)).squeeze()
                    log_lm_prob_shuf_tl = (seq_mask_y_shuf.type_as(log_lm_prob_shuf_tl) * log_lm_prob_shuf_tl).sum(dim=1)

                log_masked_lm_prob=0.0
                if masked_lm_logits is not None:
                    log_masked_lm_prob = F.log_softmax(masked_lm_logits, dim=-1)
                    log_masked_lm_prob = torch.gather(log_masked_lm_prob, 2, x_out.unsqueeze(-1)).squeeze()
                    log_masked_lm_prob = (seq_mask_x.type_as(log_masked_lm_prob) * log_masked_lm_prob * mlm_mask_positions.float() ).sum(dim=1)

                log_bow_prob=torch.zeros_like(log_lm_prob)
                log_bow_prob_tl=torch.zeros_like(log_lm_prob)

                if bow_logits is not None:
                    if model.bernoulli_bow:
                        bow_logprobs,bow_logprobs_inv =sparsedists.bernoulli.bernoulli_log_probs_from_logit(bow_logits)
                        bsz=bow_logits.size(0)
                        for i in range(bsz):
                            bow=bow_indexes[i]
                            bow_inv=bow_indexes_inv[i]
                            log_bow_prob[i]=torch.sum( bow_logprobs[i][bow] ) + torch.sum(bow_logprobs_inv[i][bow_inv])
                    else:
                        bow_logprobs=F.log_softmax(bow_logits,dim=-1)
                        #bow_logprobs_inv=torch.log(1-torch.sigmoid(bow_logits))
                        bsz=bow_logits.size(0)
                        for i in range(bsz):
                            bow=bow_indexes[i]
                            #bow_inv=bow_indexes_inv[i]
                            log_bow_prob[i]=torch.sum( bow_logprobs[i][bow] )# + torch.sum(bow_logprobs_inv[i][bow_inv])

                if bow_logits_tl is not None:
                    if model.bernoulli_bow:
                        bow_logprobs_tl,bow_logprobs_inv_tl =sparsedists.bernoulli.bernoulli_log_probs_from_logit(bow_logits)
                        bsz=bow_logits_tl.size(0)
                        for i in range(bsz):
                            bow=bow_indexes_tl[i]
                            bow_inv=bow_indexes_inv_tl[i]
                            log_bow_prob_tl[i]=torch.sum( bow_logprobs_tl[i][bow] ) + torch.sum( bow_logprobs_inv_tl[i][bow_inv] )
                    else:
                        bow_logprobs_tl=F.log_softmax(bow_logits_tl,dim=-1)
                        #bow_logprobs_inv_tl=torch.log(1-torch.sigmoid(bow_logits_tl))
                        bsz=bow_logits_tl.size(0)
                        for i in range(bsz):
                            bow=bow_indexes_tl[i]
                            #bow_inv=bow_indexes_inv_tl[i]
                            log_bow_prob_tl[i]=torch.sum( bow_logprobs_tl[i][bow] ) #+ torch.sum( bow_logprobs_inv_tl[i][bow_inv] )

                #Importance sampling: as if we were integratting over prior probabilities
                # Compute prior probability log P(z_s) and importance weight q(z_s|x)
                log_pz = pz.log_prob(z).sum(dim=1) if not model.disable_KL else 0.0 # [B, latent_size] -> [B]
                log_qz = qz.log_prob(z).sum(dim=1) if not model.disable_KL else 0.0

                # Estimate the importance weighted estimate of (the log of) P(x, y)
                batch_log_marginals[s] = log_lm_prob + log_lm_prob_tl + log_bow_prob + log_bow_prob_tl + log_lm_prob_rev + log_lm_prob_rev_tl + log_lm_prob_shuf + log_lm_prob_shuf_tl + log_masked_lm_prob*model.masked_lm_weight + log_pz - log_qz
                batch_log_marginals_lm[s] = log_lm_prob + log_pz - log_qz
                batch_log_marginals_lm_rev[s] = log_lm_prob_rev + log_pz - log_qz
                batch_log_marginals_lm_shuf[s] = log_lm_prob_shuf + log_pz - log_qz
                batch_log_marginals_masked_lm[s] = log_masked_lm_prob + log_pz - log_qz


            # Average over all samples.
            batch_log_marginal = torch.logsumexp(batch_log_marginals, dim=0) - \
                                 torch.log(torch.Tensor([n_samples]))
            batch_log_marginal_lm = torch.logsumexp(batch_log_marginals_lm, dim=0) - \
                                 torch.log(torch.Tensor([n_samples]))
            batch_log_marginals_lm_rev = torch.logsumexp(batch_log_marginals_lm_rev, dim=0) - \
                                 torch.log(torch.Tensor([n_samples]))
            batch_log_marginals_lm_shuf = torch.logsumexp(batch_log_marginals_lm_shuf, dim=0) - \
                                 torch.log(torch.Tensor([n_samples]))
            batch_log_marginals_masked_lm = torch.logsumexp(batch_log_marginals_masked_lm, dim=0) - \
                                 torch.log(torch.Tensor([n_samples]))

            log_marginal += batch_log_marginal.sum().item() # [B] -> []
            log_marginal_lm += batch_log_marginal_lm.sum().item()
            log_marginal_lm_rev += batch_log_marginals_lm_rev.sum().item()
            log_marginal_lm_shuf += batch_log_marginals_lm_rev.sum().item()
            log_marginal_masked_lm += batch_log_marginals_masked_lm.sum().item()

            num_sentences += batch_size
            num_predictions += (seq_len_x.sum() ).item()
            num_predictions_lm += (seq_len_x.sum() ).item()
            num_predictions_masked_lm += (seq_len_x.sum() ).item()*model.masked_lm_proportion
            if lm_logits_tl is not None:
                num_predictions += (seq_len_y.sum() ).item()
            if lm_rev_logits is not None:
                num_predictions += (seq_len_x_rev.sum() ).item()
            if lm_rev_logits_tl is not None:
                num_predictions += (seq_len_y_rev.sum() ).item()
            if lm_shuf_logits is not None:
                num_predictions += (seq_len_x_shuf.sum() ).item()
            if lm_shuf_logits_tl is not None:
                num_predictions += (seq_len_y_shuf.sum() ).item()
            if masked_lm_logits is not None:
                num_predictions+= (seq_len_x.sum() ).item()*model.masked_lm_proportion

            #if model.bow_output_layer is not None:
            #num_predictions+=  (model.bow_output_layer.out_features-1)*batch_size #-1 because we ignore pad_idx
            #if model.bow_output_layer_tl is not None:
            #num_predictions+= (model.bow_output_layer_tl.out_features-1)*batch_size
            num_predictions+=sum(len(bi) for bi in bow_indexes)
            num_predictions+=sum(len(bi) for bi in bow_indexes_tl)

    val_NLL = -log_marginal
    val_NLL_lm= -log_marginal_lm
    val_NLL_lm_rev = -log_marginal_lm_rev
    val_NLL_lm_shuf = -log_marginal_lm_shuf
    val_NLL_masked_lm= - log_marginal_masked_lm

    val_perplexity = np.exp(val_NLL / num_predictions)
    val_perplexity_lm = np.exp(val_NLL_lm / num_predictions_lm)
    val_perplexity_lm_rev = np.exp(val_NLL_lm_rev / num_predictions_lm)
    val_perplexity_lm_shuf = np.exp(val_NLL_lm_shuf / num_predictions_lm)
    val_perplexity_masked_lm = np.exp(val_NLL_masked_lm / num_predictions_masked_lm)
    return val_perplexity, val_NLL/num_sentences, total_KL/num_sentences, total_KL_prediction/num_sentences, val_perplexity_lm,val_NLL_lm/num_sentences, val_perplexity_lm_rev, val_NLL_lm_rev/num_sentences, val_perplexity_lm_shuf, val_NLL_lm_shuf/num_sentences, val_perplexity_masked_lm, val_NLL_masked_lm/num_sentences

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
