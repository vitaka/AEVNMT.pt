import torch
import numpy as np
import sys
import time

from aevnmt.hparams import Hyperparameters
from aevnmt.data import TextDataset, RawInputTextDataset, remove_subword_tokens, postprocess
from aevnmt.train import create_model
from aevnmt.train_utils import load_vocabularies, compute_bleu
from aevnmt.data.datasets import InputTextDataset
from aevnmt.data.textprocessing import SentenceSplitter
from aevnmt.data.textprocessing import Pipeline
from aevnmt.data.textprocessing import Tokenizer, Detokenizer
from aevnmt.data.textprocessing import Lowercaser, Truecaser, Recaser
from aevnmt.data.textprocessing import WordSegmenter, WordDesegmenter

from aevnmt import aevnmt_helper, vae_helper

from torch.utils.data import DataLoader
from pathlib import Path


class TranslationEngine:

    NBEST_SEPARATOR=" ||| "

    def __init__(self, hparams):

        output_dir = Path(hparams.output_dir)
        verbose = hparams.verbose

        if hparams.vocab_prefix is None:
            hparams.vocab_prefix = output_dir / "vocab"
            hparams.share_vocab = False

        # Select the correct device (GPU or CPU).
        device = torch.device("cuda:0") if hparams.use_gpu else torch.device("cpu")

        # Pre/post-processing
        if hparams.tokenize:
            src_tokenizer_lang = hparams.src
            tgt_tokenizer_lang = hparams.tgt
        else:
            src_tokenizer_lang = tgt_tokenizer_lang = None
        if hparams.bpe_codes_prefix:
            src_bpe_codes = f"{hparams.bpe_codes_prefix}.{hparams.src}"
            tgt_bpe_codes = f"{hparams.bpe_codes_prefix}.{hparams.tgt}"
        else:
            src_bpe_codes = tgt_bpe_codes = None

        if hparams.lowercase and hparams.truecaser_prefix:
            raise ValueError("You cannot use lowercasing and truecasing at the same time")

        if hparams.truecaser_prefix:
            src_truecase_model = f"{hparams.truecaser_prefix}.{hparams.src}"
            tgt_truecase_model = f"{hparams.truecaser_prefix}.{hparams.tgt}"
        else:
            src_truecase_model = tgt_truecase_model = None

        model_checkpoint = output_dir / f"model/{hparams.criterion}/{hparams.src}-{hparams.tgt}.pt"

        self.hparams = hparams
        self.verbose = verbose
        self.device = device
        self.model_checkpoint = model_checkpoint
        self.src_tokenizer_lang = src_tokenizer_lang
        self.tgt_tokenizer_lang = tgt_tokenizer_lang
        self.src_bpe_codes = src_bpe_codes
        self.tgt_bpe_codes = tgt_bpe_codes
        self.src_truecase_model = src_truecase_model
        self.tgt_truecase_model = tgt_truecase_model
        self.pipeline = Pipeline()
        self.vocab_src = None
        self.vocab_tgt = None
        self.model = None
        self.translate_fn = None
        self.n_translated = 0

    @staticmethod
    def make_pipeline(hparams):
        # Loading pre/post-processing models
        if hparams.verbose:
            print("Loading pre/post-processing models", file=sys.stderr)

        preprocess = []
        postprocess = []

        # Tokenization
        if hparams.tokenize:
            preprocess.append(Tokenizer(hparams.src))
        if hparams.detokenize:
            postprocess.append(Detokenizer(hparams.tgt if not hparams.re_generate_sl else hparams.src))

        # Case
        if hparams.lowercase and hparams.truecaser_prefix:
            raise ValueError("You cannot set --lowercase to true and provide a --truecaser_prefix at the same time")

        if hparams.lowercase:
            preprocess.append(Lowercaser(hparams.src))

        if hparams.truecaser_prefix:
            preprocess.append(Truecaser(f"{hparams.truecaser_prefix}.{hparams.src}"))
        if hparams.recase:
            postprocess.append(Recaser(hparams.tgt if not hparams.re_generate_sl else hparams.src))

        # Word segmentation
        if hparams.bpe_codes_prefix:
            preprocess.append(WordSegmenter(f"{hparams.bpe_codes_prefix}.{hparams.src}", separator=hparams.subword_token))
        if hparams.bpe_merge:
            postprocess.append(WordDesegmenter(separator=hparams.subword_token))

        return Pipeline(pre=preprocess, post=list(reversed(postprocess)))

    def load_statics(self):
        # Loading vocabulary
        if self.verbose:
            t0 = time.time()
            print(f"Loading vocabularies src={self.hparams.src} tgt={self.hparams.tgt}", file=sys.stderr)
        self.vocab_src, self.vocab_tgt = load_vocabularies(self.hparams)

        # Load pre/post processing models and configure a pipeline
        self.pipeline = TranslationEngine.make_pipeline(self.hparams)

        if self.verbose:
            print(f"Restoring model selected wrt {self.hparams.criterion} from {self.model_checkpoint}", file=sys.stderr)

        model, _, _, translate_fn = create_model(self.hparams, self.vocab_src, self.vocab_tgt)
        if self.hparams.use_gpu:
            tdict=torch.load(self.model_checkpoint)
            newkeys={}
            for k in tdict:
                if k.startswith("inf_network."):
                    predk=k.replace("inf_network.","pred_network.")
                    if not predk in tdict:
                        newkeys[predk]=tdict[k]
            tdict.update(newkeys)
            model.load_state_dict(tdict)
        else:
            model.load_state_dict(torch.load(self.model_checkpoint, map_location='cpu'))

        self.model = model.to(self.device)
        self.translate_fn = translate_fn
        if self.hparams.re_generate_sl or self.hparams.re_generate_tl:
            if self.hparams.model_type == "aevnmt":
                self.translate_fn=aevnmt_helper.re_sample
            elif self.hparams.model_type == "vae":
                self.translate_fn=vae_helper.re_sample
            else:
                raise NotImplementedError
        self.model.eval()
        if self.verbose:
            print("Done loading in %.2f seconds" % (time.time() - t0), file=sys.stderr)

    def translate(self, lines: list ,  stdout=sys.stdout, z_lines=None, y_lines=None):
        hparams = self.hparams
        if hparams.split_sentences:  # This is a type of pre-processing we do not a post-processing counterpart for
            if hparams.verbose:
                print(f"Running sentence splitter for {len(lines)} lines")
            lines = SentenceSplitter(hparams.src).split(lines)
            if hparams.verbose:
                print(f"Produced {len(lines)} sentences")
        if not lines:  # we do not like empty jobs
            return []
        aux_data_generator=None
        #assert not (z_lines is not None and y_lines is not None)
        if z_lines is not None:
            aux_data_generator=({'z':eval(l)} for l in z_lines)
        elif y_lines is not None:
            aux_data_generator=({'y': l.rstrip("\n")} for l in y_lines)
        input_data = InputTextDataset(
            generator=(self.pipeline.pre(line) for line in lines),
            max_length=hparams.max_sentence_length,
            split=True,
            aux_data_generator=aux_data_generator)
        input_dl = DataLoader(
            input_data, batch_size=hparams.batch_size,
            shuffle=False, num_workers=4)
        input_size = len(input_data)

        # Translate the data.
        num_translated = 0
        all_hypotheses = []
        all_zs=[]
        if self.verbose:
            print(f"Translating {input_size} sentences...", file=sys.stderr)

        for input_sentences in input_dl:

            input_zs=None
            if 'z' in input_sentences:
                #For some reason, Torch DataLoader transposed the input zs
                #and we need to restore to (batch_size, z_dim) size
                input_zs=torch.stack(input_sentences['z'],dim=0).transpose(1,0)

            input_ys=None
            if 'y' in input_sentences:
                #import pdb; pdb.set_trace()
                input_ys=input_sentences['y']

            # Sort the input sentences from long to short.
            input_sentences = np.array(input_sentences['sentence'])
            seq_len = np.array([len(s.split()) for s in input_sentences])
            sort_keys = np.argsort(-seq_len)
            input_sentences = input_sentences[sort_keys]
            if input_zs is not None:
                input_zs=input_zs[sort_keys]

            if input_ys is not None:
                input_ys=np.array(input_ys)[sort_keys]

            t1 = time.time()
            # Translate the sentences using the trained model.
            moreargs={}
            if hparams.sample_posterior_decoding or hparams.sample_prior_decoding:
                moreargs['deterministic']=False
            if input_zs is not None:
                moreargs['z']=input_zs.float()
            if input_ys is not None:
                moreargs['input_sentences_y']=input_ys
            if hparams.sample_prior_decoding:
                moreargs['use_prior']=True
            if hparams.re_generate_tl:
                moreargs['use_tl_lm']=True
            hypotheses,zs = self.translate_fn(
                self.model, input_sentences,
                self.vocab_src, self.vocab_tgt,
                self.device, hparams, **moreargs)

            num_translated += len(input_sentences)

            # Restore the original ordering.
            inverse_sort_keys = np.argsort(sort_keys)
            all_hypotheses += hypotheses[inverse_sort_keys].tolist()
            if zs is not None:
                all_zs+=zs[inverse_sort_keys].tolist()

            if self.verbose:
                print(f"{num_translated}/{input_size} sentences translated in {time.time() - t1:.2f} seconds.", file=sys.stderr)

        if hparams.show_raw_output:
            for i in range(len(input_data)):
                print(i + self.n_translated, '|||', input_data[i], '|||', all_hypotheses[i][0], file=sys.stderr)

        if hparams.max_sentence_length > 0:  # join sentences that might have been split
            all_hypotheses =  np.array([ input_data.join(hyps_n) for hyps_n in np.array(all_hypotheses).transpose(1,0)  ]).transpose(1,0)

        # Post-processing
        all_hypotheses = [ [self.pipeline.post(h) for h in h_nbest] for h_nbest in all_hypotheses]

        if stdout is not None:
            for nbest_hypotheses in all_hypotheses:
                print(self.NBEST_SEPARATOR.join(nbest_hypotheses) , file=stdout)

        self.n_translated += len(input_data)

        return all_hypotheses,all_zs

    def interactive_translation_n(self, generator=sys.stdin, wait_for=1, stdout=sys.stdout):
        if self.verbose:
            print(f"Ready to start translating {wait_for} sentences at a time", file=sys.stderr)
        job = []
        for line in generator:
            job.append(line)
            if len(job) >= wait_for:
                self.translate(job, stdout=stdout)
                job = []
            if self.verbose:
                print(f"Waiting for {wait_for - len(job)} sentences", file=sys.stderr)

    def interactive_translation(self, generator=sys.stdin, stdout=sys.stdout):
        if self.verbose:
            print("Ready to start", file=sys.stderr)
        for i, line in enumerate(generator):
            self.translate([line], stdout=stdout)

    def translate_file(self, input_path, output_path=None, reference_path=None, stdout=None, output_z_path=None, input_z_path=None, input_y_path=None):
        if output_path is None:
            stdout = sys.stdout

        z_lines=None
        if input_z_path is not None:
            with open(input_z_path) as z_f:
                z_lines=z_f.readlines()
        y_lines=None
        if input_y_path is not None:
            with open(input_y_path) as y_f:
                y_lines=y_f.readlines()

        with open(input_path) as f:  # TODO: optionally segment input file into slices of n lines each
            translations,zs = self.translate(f.readlines(), stdout=stdout,z_lines=z_lines,y_lines=y_lines)
            # If a reference set is given compute BLEU score.
            if reference_path is not None:
                ref_sentences = TextDataset(reference_path).data
                if self.hparams.postprocess_ref:
                    ref_sentences = [self.pipeline.post(r) for r in ref_sentences]
                bleu = compute_bleu([ t_nbest[0] for t_nbest in translations], ref_sentences, subword_token=None)
                print(f"\nBLEU = {bleu:.4f}")

            # If an output file is given write the output to that file.
            if output_path is not None:
                if self.verbose:
                    print(f"\nWriting translation output to {output_path}", file=sys.stderr)
                with open(output_path, "w") as f:
                    for translation_nbest in translations:
                        f.write("{}\n".format(self.NBEST_SEPARATOR.join(translation_nbest)))

            if output_z_path is not None:
                if self.verbose:
                    print(f"\nWriting zs to {output_z_path}", file=sys.stderr)
                with open(output_z_path, "w") as f:
                    for z in zs:
                        f.write("{}\n".format(z))


    def translate_stdin(self, stdout=sys.stdout):
        lines = [line for line in sys.stdin]
        self.translate(lines, stdout=stdout)


def main(hparams=None):
    # Load command line hyperparameters (and if provided from an hparams_file).
    if hparams is None:
        hparams = Hyperparameters(check_required=False)
        # Fill in any missing values from the hparams file in the output_dir.
        output_dir = Path(hparams.output_dir)
        hparams_file = output_dir / "hparams"
        hparams.update_from_file(hparams_file, override=False)

    engine = TranslationEngine(hparams)

    engine.load_statics()

    if hparams.interactive_translation > 0:
        if hparams.interactive_translation == 1:
            engine.interactive_translation()
        else:
            engine.interactive_translation_n(wait_for=hparams.interactive_translation)
    elif hparams.translation_input_file == '-':
        engine.translate_stdin()
    else:
        if hparams.translation_ref_file and hparams.split_sentences:
            raise ValueError("If you enable sentence splitting you will compromise line-alignment with the reference")
        if hparams.max_sentence_length > -1 and hparams.z_input_file:
            raise ValueError("If you enable sentence splitting you will compromise line-alignment with the input zs")
        engine.translate_file(
            input_path=hparams.translation_input_file,
            output_path=hparams.translation_output_file,
            reference_path=hparams.translation_ref_file,
            output_z_path=hparams.z_output_file,
            input_z_path=hparams.z_input_file,
            input_y_path=hparams.y_input_file
        )

if __name__ == "__main__":
    main()
