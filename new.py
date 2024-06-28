# !pip install TTS numpy==1.23.5

import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
from dataclasses import dataclass, field

from trainer import Trainer, TrainerArgs

from TTS.config import load_config, register_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model



@dataclass
class TrainTTSArgs(TrainerArgs):
    config_path: str = field(default=None, metadata={"help": "Path to the config file."})

def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = "/kaggle/working/tts/MyTTSDataset/transcript (1).txt"
    items = []
    speaker_name = "my_speaker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("@")
            path = "/kaggle/input/tts-hindi-f/Hindi-F/wav/"
            wav_file = f"{path}{cols[0]}.wav"
            text = cols[1]
            # print(text)
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items

def main():
    """Run `tts` model training directly by a `config.json` file."""
    # init trainer args
    train_args = TrainTTSArgs()
    parser = train_args.init_argparse(arg_prefix="")

    # override trainer args from comman-line args
    args, config_overrides = parser.parse_known_args()
    train_args.parse_args(args)
    

    # load config.json and register
    args.continue_path = "/kaggle/working/tts/Hindi-TTS-01-June-27-2024_03+00PM-0000000"
    
    if args.config_path or args.continue_path:
        if args.config_path:
            # init from a file
            config = load_config(args.config_path)
            if len(config_overrides) > 0:
                config.parse_known_args(config_overrides, relaxed_parser=True)
        elif args.continue_path:
            # continue from a prev experiment
            config = load_config(os.path.join(args.continue_path, "config.json"))
            if len(config_overrides) > 0:
                config.parse_known_args(config_overrides, relaxed_parser=True)
        else:
            # init from console args
            from TTS.config.shared_configs import BaseTrainingConfig  # pylint: disable=import-outside-toplevel

            config_base = BaseTrainingConfig()
            config_base.parse_known_args(config_overrides)
            config = register_config(config_base.model)()

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
        formatter=formatter
    )

    # init the model from config
    model = setup_model(config, train_samples + eval_samples)

    # init the trainer and 🚀
    trainer = Trainer(
        train_args,
        model.config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    trainer.fit()


if __name__ == "__main__":
    main()





































###########################################################################################################################################
# # %% [code] {"id":"ZfpdcfN0E7VP","outputId":"02c6d529-cebc-4e1f-9686-c27d41b1d2a1","jupyter":{"outputs_hidden":false}}
# !git clone https://github.com/coqui-ai/TTS
# !pip install -e TTS/[all,dev,notebooks]

# # %% [code] {"id":"FOQspBKvsIfK","jupyter":{"outputs_hidden":false}}
# # !pip install -U pip -q
# # !pip install TTS -q

# !mkdir tts
# !mkdir tts/MyTTSDataset
# !cp '/kaggle/input/tts-hindi-f/transcript (1).txt' tts/MyTTSDataset/
# !mkdir tts/MyTTSDataset/wavs
# !cp /kaggle/input/tts-hindi-f/Hindi-F/wav/* tts/MyTTSDataset/wavs/

# # %% [markdown] {"id":"LdU0k0Omuaxi"}
# # Import Dataset

# # %% [code] {"jupyter":{"outputs_hidden":false}}
# !pip install numpy

# # %% [code] {"id":"hKuDYZbxuaJO","outputId":"3622acc0-4fd0-4b43-c101-6f8807daf1b0","jupyter":{"outputs_hidden":false}}
# # import json

# # config_metadata= {"username":"skywalker290","key":"e2facce0e3e784ef3acbd318fdce066b"}

# # with open('kaggle.json', 'w') as f:
# #     json.dump(config_metadata, f)

# # !mkdir -p ~/.kaggle
# # !mv kaggle.json ~/.kaggle/
# # !chmod 600 ~/.kaggle/kaggle.json


# # !kaggle datasets download -d skywalker290/hindi-m
# # !unzip hindi-m.zip &> /dev/null
# # !rm hindi-m.zip

# # %% [code] {"id":"NTuY6pIbsa8U","outputId":"b4f47d91-c5c5-45ac-d03d-794ca2317e45","jupyter":{"outputs_hidden":false}}
# # from google.colab import drive
# # drive.mount('/content/gdrive')

# # %% [markdown] {"id":"DaCAG17etqJo"}
# # We are importing the files

# # %% [code] {"id":"ziS7GT99Owl0","outputId":"764d3734-c211-49cb-d808-6a3a3941e8d0","jupyter":{"outputs_hidden":false}}
# # !pip install numpy==1.23
# # !pip show numpy

# # %% [code] {"id":"NuoRBPqPtRrm","execution":{"iopub.status.busy":"2024-06-27T14:57:38.085473Z","iopub.execute_input":"2024-06-27T14:57:38.086163Z","iopub.status.idle":"2024-06-27T14:57:49.808538Z","shell.execute_reply.started":"2024-06-27T14:57:38.086123Z","shell.execute_reply":"2024-06-27T14:57:49.807752Z"},"jupyter":{"outputs_hidden":false}}
# import os

# from trainer import Trainer, TrainerArgs

# from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
# from TTS.tts.configs.vits_config import VitsConfig
# from TTS.tts.datasets import load_tts_samples
# from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
# from TTS.tts.utils.speakers import SpeakerManager
# from TTS.tts.utils.text.tokenizer import TTSTokenizer
# from TTS.utils.audio import AudioProcessor

# # %% [markdown] {"id":"lutgNmiGzZuo"}
# # Generate Transcript file

# # %% [code] {"id":"f3IKgUdBzYgr","outputId":"cd636854-9f16-4340-de4c-73674feea24b","jupyter":{"outputs_hidden":false}}
# # import os
# # from tqdm import tqdm

# # def generate_transcript(wav_dir, txt_dir, output_file):
# #     transcript_entries = []

# #     wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

# #     # Use tqdm to add a progress bar
# #     for wav_filename in tqdm(wav_files, desc="Processing files"):
# #         base_filename = os.path.splitext(wav_filename)[0]
# #         txt_filename = f"{base_filename}.txt"
# #         txt_filepath = os.path.join(txt_dir, txt_filename)

# #         # Read the corresponding txt file
# #         if os.path.exists(txt_filepath):
# #             with open(txt_filepath, 'r', encoding='utf-8') as txt_file:
# #                 transcript_text = txt_file.read().strip()
# #                 transcript_entry = f"{wav_filename.split('.')[0]}@{transcript_text}"
# #                 transcript_entries.append(transcript_entry)
# #         else:
# #             print(f"Warning: No transcript found for {wav_filename}")

# #     # Write to the output transcript file
# #     with open(output_file, 'w', encoding='utf-8') as output:
# #         for entry in transcript_entries:
# #             output.write(entry + '\n')

# # # Specify the directories and the output file
# # wav_directory = "/path/to/wav/files"
# # txt_directory = "/path/to/txt/files"
# # output_transcript_file = "/path/to/output/transcript.txt"

# # # Generate the transcript.txt file
# # generate_transcript("/content/Hindi_M/wav", "/content/Hindi_M/txt", "/content/Hindi_M/transcript.txt")

# # print(f"Transcript file generated at: {output_transcript_file}")

# # %% [code] {"id":"kXg_lDM-0ont","jupyter":{"outputs_hidden":false}}
# # !wc /content/Hindi_M/transcript.txt

# # %% [markdown] {"id":"stAeMrxY3wm3"}
# # Preparing the TTS formated dataset

# # %% [code] {"id":"nTGLPMwq3vvf","outputId":"d035da66-5a00-44b2-9a71-4744f801ba06","jupyter":{"outputs_hidden":false}}


# # %% [code] {"id":"3uTeTYgWtvl9","execution":{"iopub.status.busy":"2024-06-27T14:57:56.680458Z","iopub.execute_input":"2024-06-27T14:57:56.680825Z","iopub.status.idle":"2024-06-27T14:57:56.686115Z","shell.execute_reply.started":"2024-06-27T14:57:56.680796Z","shell.execute_reply":"2024-06-27T14:57:56.685089Z"},"jupyter":{"outputs_hidden":false}}
# output_path = "/kaggle/working/tts"# path to the dataset

# dataset_config = BaseDatasetConfig(
#     formatter="ljspeech", meta_file_train="transcript (1).txt", path=os.path.join(output_path, "MyTTSDataset/")
# )

# # %% [markdown] {"id":"5dAjTTOL4BHO"}
# # 
# # Audio Config

# # %% [code] {"id":"3pcPWScC4ARp","execution":{"iopub.status.busy":"2024-06-27T14:57:58.909364Z","iopub.execute_input":"2024-06-27T14:57:58.910323Z","iopub.status.idle":"2024-06-27T14:57:58.918059Z","shell.execute_reply.started":"2024-06-27T14:57:58.910275Z","shell.execute_reply":"2024-06-27T14:57:58.915461Z"},"jupyter":{"outputs_hidden":false}}
# audio_config = VitsAudioConfig(
#     sample_rate=44100, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
# )

# # %% [markdown] {"id":"CqbOX7O6ybQy"}
# # Functions to generate characters for our dataset

# # %% [code] {"id":"QJ8OxrgSyZmX","outputId":"efe0bdf0-405a-441d-9b47-75d0a81925a5","execution":{"iopub.status.busy":"2024-06-27T14:58:04.533786Z","iopub.execute_input":"2024-06-27T14:58:04.534640Z","iopub.status.idle":"2024-06-27T14:58:12.589790Z","shell.execute_reply.started":"2024-06-27T14:58:04.534606Z","shell.execute_reply":"2024-06-27T14:58:12.588714Z"},"jupyter":{"outputs_hidden":false}}
# import os
# import regex

# # Function to get all Devanagari characters
# def get_devanagari_characters():
#     devanagari_characters = set()
#     for codepoint in range(0x0900, 0x097F + 1):
#         character = chr(codepoint)
#         if regex.match(r'\p{Devanagari}', character):
#             devanagari_characters.add(character)
#     return devanagari_characters

# # Function to traverse .txt files and extract unique characters and punctuations
# def get_unique_characters_and_punctuations(directory):
#     devanagari_characters = get_devanagari_characters()
#     unique_characters = set()
#     unique_punctuations = set()

#     # Define a set of common punctuation marks
#     common_punctuations = set(" !,.?-।")

#     for filename in os.listdir(directory):
#         if filename.endswith(".txt"):
#             with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
#                 text = file.read()
#                 for char in text:
#                     if char in common_punctuations:
#                         unique_punctuations.add(char)
#                     elif char in devanagari_characters:
#                         unique_characters.add(char)

#     return unique_characters, unique_punctuations

# # Function to create the CharactersConfig
# def create_characters_config(directory):
#     unique_characters, unique_punctuations = get_unique_characters_and_punctuations(directory)

#     characters = "".join(sorted(unique_characters))
#     punctuations = "".join(sorted(unique_punctuations))

#     return {'chars':characters,'punc':punctuations}

# # Specify the directory containing the .txt files
# directory = "/kaggle/input/tts-hindi-f/Hindi-F/txt"
# chars_punc = create_characters_config(directory)

# # Print the character config
# print("Characters Config:")
# print(chars_punc)

# # %% [code] {"id":"V0Mh_SIu4HLY","execution":{"iopub.status.busy":"2024-06-27T14:58:12.591306Z","iopub.execute_input":"2024-06-27T14:58:12.591818Z","iopub.status.idle":"2024-06-27T14:58:12.596894Z","shell.execute_reply.started":"2024-06-27T14:58:12.591787Z","shell.execute_reply":"2024-06-27T14:58:12.595953Z"},"jupyter":{"outputs_hidden":false}}
# character_config = CharactersConfig(
#     characters_class= "TTS.tts.models.vits.VitsCharacters",
#     characters= chars_punc['chars'],
#     punctuations= chars_punc['punc'],
#     pad= "<PAD>",
#     eos= "<EOS>",
#     bos= "<BOS>",
#     blank= "<BLNK>",
# )

# # %% [code] {"id":"3sOF7zwu4vwe","outputId":"05ef4b59-38c2-48d4-dae0-053b240d3c75","execution":{"iopub.status.busy":"2024-06-27T14:58:27.805234Z","iopub.execute_input":"2024-06-27T14:58:27.805658Z","iopub.status.idle":"2024-06-27T14:58:27.810760Z","shell.execute_reply.started":"2024-06-27T14:58:27.805628Z","shell.execute_reply":"2024-06-27T14:58:27.809802Z"},"jupyter":{"outputs_hidden":false}}
# print(character_config)

# # %% [markdown] {"id":"0-j0F4yn5MoJ"}
# # Model Configuration

# # %% [code] {"id":"pcXPjh5P4yQA","execution":{"iopub.status.busy":"2024-06-27T14:58:29.004601Z","iopub.execute_input":"2024-06-27T14:58:29.004940Z","iopub.status.idle":"2024-06-27T14:58:29.014030Z","shell.execute_reply.started":"2024-06-27T14:58:29.004917Z","shell.execute_reply":"2024-06-27T14:58:29.012927Z"},"jupyter":{"outputs_hidden":false}}
# config = VitsConfig(
#     audio=audio_config,
#     characters=character_config,
#     run_name="Hindi-TTS-01",
#     batch_size=16,
#     eval_batch_size=4,
#     num_loader_workers=4,
#     num_eval_loader_workers=4,
#     run_eval=True,
#     test_delay_epochs=0,
#     epochs=1000,
#     text_cleaner="basic_cleaners",
#     use_phonemes=False,
#     phoneme_language="en-us",
#     phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
#     compute_input_seq_cache=True,
#     print_step=25,
#     print_eval=False,
#     save_best_after=1000,
#     save_checkpoints=True,
#     save_all_best=True,
#     mixed_precision=True,
#     max_text_len=250,  # change this if you have a larger VRAM than 16GB
#     output_path=output_path,
#     datasets=[dataset_config],
#     cudnn_benchmark=False,
#     test_sentences=[
#         ["मेरा नाम क्या है"],
#         ["हेलो बरोथेर और क्या चल रहा"],
#         ["ईशा अल्लाह लड़कों ने अच्छा खेला"]
#     ]
# )

# # %% [markdown] {"id":"96XcaoIf57Vk"}
# # seting up tokeniser

# # %% [code] {"id":"9wMuIrWN5xME","outputId":"018de5b9-e961-4ead-ac6f-8511fea67f29","execution":{"iopub.status.busy":"2024-06-27T14:58:36.304226Z","iopub.execute_input":"2024-06-27T14:58:36.304620Z","iopub.status.idle":"2024-06-27T14:58:36.314093Z","shell.execute_reply.started":"2024-06-27T14:58:36.304590Z","shell.execute_reply":"2024-06-27T14:58:36.313139Z"},"jupyter":{"outputs_hidden":false}}
# # Audio processor is used for feature extraction and audio I/O.
# # It mainly serves to the dataloader and the training loggers.
# ap = AudioProcessor.init_from_config(config)

# # INITIALIZE THE TOKENIZER
# # Tokenizer is used to convert text to sequences of token IDs.
# # config is updated with the default characters if not defined in the config.
# tokenizer, config = TTSTokenizer.init_from_config(config)

# # %% [markdown] {"id":"YNmgxOPY6AKV"}
# # formatter function

# # %% [code] {"id":"oqvfMoNM6cLr","execution":{"iopub.status.busy":"2024-06-27T15:00:16.431834Z","iopub.execute_input":"2024-06-27T15:00:16.432221Z","iopub.status.idle":"2024-06-27T15:00:16.436690Z","shell.execute_reply.started":"2024-06-27T15:00:16.432193Z","shell.execute_reply":"2024-06-27T15:00:16.435560Z"},"jupyter":{"outputs_hidden":false}}
# path_wav='/kaggle/input/tts-hindi-f/Hindi-F/wav/'

# # %% [code] {"id":"1hiaG6_w5_es","execution":{"iopub.status.busy":"2024-06-27T15:00:17.029230Z","iopub.execute_input":"2024-06-27T15:00:17.030315Z","iopub.status.idle":"2024-06-27T15:00:17.036551Z","shell.execute_reply.started":"2024-06-27T15:00:17.030280Z","shell.execute_reply":"2024-06-27T15:00:17.035624Z"},"jupyter":{"outputs_hidden":false}}
# def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
#     """Assumes each line as ```<filename>|<transcription>```
#     """
#     txt_file = "/kaggle/working/tts/MyTTSDataset/transcript (1).txt"
#     items = []
#     speaker_name = "my_speaker"
#     with open(txt_file, "r", encoding="utf-8") as ttf:
#         for line in ttf:
#             cols = line.split("@")
#             wav_file = f"{path_wav}{cols[0]}.wav"
#             text = cols[1]
#             # print(text)
#             items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
#     return items

# # %% [code] {"jupyter":{"outputs_hidden":false}}


# # %% [code] {"id":"8xecHOXP6oVi","outputId":"f8541abc-7335-4f08-9d5a-6f7bc76483c8","execution":{"iopub.status.busy":"2024-06-27T15:00:21.684033Z","iopub.execute_input":"2024-06-27T15:00:21.684699Z","iopub.status.idle":"2024-06-27T15:00:22.114356Z","shell.execute_reply.started":"2024-06-27T15:00:21.684668Z","shell.execute_reply":"2024-06-27T15:00:22.113416Z"},"jupyter":{"outputs_hidden":false}}
# train_samples, eval_samples = load_tts_samples(
# dataset_config,
# eval_split=True,
# formatter=formatter)

# # %% [code] {"id":"_Qte8ZNG7wwb","outputId":"f9923a42-5e1a-4090-92f8-01d9691e3244","jupyter":{"outputs_hidden":false}}
# def validate_transcript(transcript_file):
#     with open(transcript_file, 'r', encoding='utf-8') as file:
#         lines = file.readlines()

#     valid_entries = []
#     invalid_entries = []

#     for line in lines:
#         parts = line.strip().split('@')
#         if len(parts) >= 2 and parts[0] and parts[1]:
#             valid_entries.append(line.strip())
#         else:
#             invalid_entries.append(line.strip())

#     return valid_entries, invalid_entries

# # Specify the path to the transcript.txt file
# transcript_file_path = "/content/tts/MyTTSDataset/transcript.txt"

# # Validate the transcript file
# valid_entries, invalid_entries = validate_transcript(transcript_file_path)

# # Print the results
# # print("Valid Entries:")
# # for entry in valid_entries:
# #     print(entry)

# print("\nInvalid Entries:")
# for entry in invalid_entries:
#     print(entry)

# # %% [code] {"id":"jd1Oo2rZ8Xyk","outputId":"f833c94d-d0ae-49d2-c294-5ed287674f25","jupyter":{"outputs_hidden":false}}
# valid_entries[0]

# # %% [code] {"id":"mHNt_8mu-zCK","jupyter":{"outputs_hidden":false}}
# def create_valid_transcript(valid_entries, output_file):
#     with open(output_file, 'w', encoding='utf-8') as file:
#         for entry in valid_entries:
#             file.write(entry + '\n')

# create_valid_transcript(valid_entries, "/content/tts/MyTTSDataset/transcript.txt")

# # %% [code] {"jupyter":{"outputs_hidden":false}}
# !pip show tensorflow

# # %% [code] {"jupyter":{"outputs_hidden":false}}
# !pip install numpy==1.23.5

# # %% [code] {"id":"eTYVsOHV_dlq","outputId":"09703fdd-5f83-440e-d1b4-6f6aeb557393","execution":{"iopub.status.busy":"2024-06-27T15:00:26.215440Z","iopub.execute_input":"2024-06-27T15:00:26.215893Z","iopub.status.idle":"2024-06-27T15:00:27.679008Z","shell.execute_reply.started":"2024-06-27T15:00:26.215858Z","shell.execute_reply":"2024-06-27T15:00:27.678086Z"},"jupyter":{"outputs_hidden":false}}
# # init model
# model = Vits(config, ap, tokenizer, speaker_manager=None)

# # init the trainer and 🚀
# trainer = Trainer(
#     TrainerArgs(),
#     config,
#     output_path,
#     model=model,
#     train_samples=train_samples,
#     eval_samples=eval_samples,
# )

# # %% [code] {"id":"3UtF4rgED1TB","outputId":"448a879a-375e-4119-d9bc-9246728ac3b9","execution":{"iopub.status.busy":"2024-06-27T15:00:30.119486Z","iopub.execute_input":"2024-06-27T15:00:30.119838Z"},"jupyter":{"outputs_hidden":false}}
# trainer.fit()

# # %% [code] {"id":"SqFEgcXFPZSB","jupyter":{"outputs_hidden":false}}
# # !rm /kaggle/working/tts/MyTTSDataset/wavs/* -rf

# # %% [code] {"jupyter":{"outputs_hidden":false}}
