import torch
import speech_recognition as sr
from typing import Optional

import sys
sys.path.append('../')

from whisper_mic import WhisperMic

import os
os.environ['OPENAI_API_KEY'] = '[ここに自分のAPIキー]'   #OpenAIのAPIキーを入力

import openai

class DialogueSystem():
    def __init__(self, init_prompt: str, lang_model: str, whisper_model: str, english: bool, verbose: bool, energy: int, pause: float, dynamic_energy: bool, save_file: bool, device: str, loop: bool, dictate: bool,mic_index:Optional[int],list_devices: bool,faster: bool,hallucinate_threshold:int) -> None:
        self.instract_prompt = init_prompt
        self.user_persona = []
        self.system_persona = []  #listにしたのはペルソナの数の管理が楽になるから
        self.system_response = ""
        self.mic = WhisperMic(model=whisper_model, english=english, verbose=verbose, energy=energy, pause=pause, dynamic_energy=dynamic_energy, save_file=save_file, device=device,mic_index=mic_index,implementation=("faster_whisper" if faster else "whisper"),hallucinate_threshold=hallucinate_threshold)
        self.lang_model = lang_model
        self.dialogue_history = []
        
        #ここら辺は正直コンストラクタになくてもいいもの。今後の拡張可能性を考慮して一応入れている。
        self.english = english  
        self.verbose = verbose
        self.energy = energy
        self.pause = pause
        self.dynamic_energy = dynamic_energy
        self.save_file = save_file
        self.device = device
        self.loop = loop 
    

    def response(self, user_dialog):
        user_message = {"role": "user", "content": user_dialog}
        
        self.dialogue_history.append(user_message)

        response = openai.ChatCompletion.create(
            model = self.lang_model,
            messages = [{"role": "system", "content": self.instract_prompt + "[対話者の特徴・ペルソナ]\n" +\
                        ''.join(self.user_persona) + "\n[あなたの特徴・ペルソナ]\n" + "".join(self.system_persona)\
                        +"それでは対話を開始します\n"}] + self.dialogue_history,
            max_tokens = 256
        )

        print(response["choices"][0]["message"]["content"])

        if len(self.dialogue_history) >= 20: 
            del self.dialogue_history[0]

        return "generate response", response["choices"][0]["message"]["content"]
    

    def speech_recognition(self):
        try:
            while True:
                result = self.mic.listen()
                print(result)
                if "BLANK_AUDIO" in result:
                    continue
                break
        except KeyboardInterrupt:
            print("Operation interrupted successfully")
        finally:
            if self.save_file:
                self.mic.file.close()

        return result


    def extract_persona(self, message):
        instract_message = "発話から発話者のペルソナを箇条書きで抽出してください。\
                            出力するのは箇条書きの部分だけです。\
                            もし類似しているものや矛盾しているものがあればどちらかを出力してください。\
                            目立った特徴・ペルソナがなければ、[None]とだけ出力して下さい。\
                            以下が目的の文です。\n"
         
        extracted_persona = openai.ChatCompletion.create(
            model = self.lang_model,
            messages = [{"role": "system", "content": instract_message + message}],
            max_tokens = 256
        )

        return "extract persona", extracted_persona["choices"][0]["message"]["content"]    

        
     