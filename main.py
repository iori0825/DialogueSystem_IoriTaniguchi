import click
import concurrent.futures
import torch
from typing import Optional

from DialogueSystem.DialogueSystem import DialogueSystem

#コマンドラインに入れられる引数(使わないのもあるけど一応入れておく)
@click.command()
@click.option("--langmodel", default="gpt-3.5-turbo", help="Model to use", type=click.Choice(["gpt-3.5-turbo","gpt-3.5-turbo-16k","gpt-4","gpt-4-turbo-preview"]))
@click.option("--whispermodel", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large","large-v2","large-v3"]))
@click.option("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use", type=click.Choice(["cpu","cuda","mps"]))
@click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--save_file",default=False, help="Flag to save file", is_flag=True,type=bool)
@click.option("--loop", default=False, help="Flag to loop", is_flag=True,type=bool)
@click.option("--dictate", default=False, help="Flag to dictate (implies loop)", is_flag=True,type=bool)
@click.option("--mic_index", default=None, help="Mic index to use", type=int)
@click.option("--list_devices",default=False, help="Flag to list devices", is_flag=True,type=bool)
@click.option("--faster",default=False, help="Use faster_whisper implementation", is_flag=True,type=bool)
@click.option("--hallucinate_threshold",default=400, help="Raise this to reduce hallucinations.  Lower this to activate more often.", is_flag=True,type=int)



def main(langmodel: str, whispermodel: str, english: bool, verbose: bool, energy: int, pause: float, dynamic_energy: bool, save_file: bool, device: str, loop: bool, dictate: bool, mic_index:Optional[int], list_devices: bool, faster: bool, hallucinate_threshold:int) -> None:
    
    init_prompt = "あなたは対話者の友達です。これからの対話はフレンドリーに接してください。できるだけ簡潔に、最大でも100文字以内で返答してください。\
                    また、あなたにはあなたのペルソナとユーザのペルソナがそれぞれ与えられています。対話はそれに基づいて行ってください。\
                    もしペルソナになければ、ペルソナに依らず、文脈に沿って自由に発話してください。以下に対話者のペルソナ([対話者の特徴・ペルソナ])と自分のペルソナ([あなたの特徴・ペルソナ]を貼ります。)\
                    "
    dialogue_system = DialogueSystem(init_prompt,langmodel, whispermodel, english, verbose, energy, pause, dynamic_energy, save_file, device, loop, dictate, mic_index, list_devices, faster, hallucinate_threshold)

    print("対話を終了したいときは「さようなら」といってください。")

    while True:
        usr_uttr = dialogue_system.speech_recognition()
        #print(usr_uttr + "\n")
        if usr_uttr == "さようなら":
            print("またお話ししましょう！")
            exit()
        
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(dialogue_system.response, usr_uttr)
            futures.append(future)
            future = executor.submit(dialogue_system.extract_persona, usr_uttr)
            futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if "generate response" in result[0]:
                    dialogue_system.system_response = result[1]
                    dialogue_system.dialogue_history.append({"role": "system", "content": result[1]})
                elif "extract persona" in result[0]:
                    if not "[None]" in result[1]: #Noneがかえって来なければ
                        dialogue_system.user_persona += result[1].split("\n")
                        if len(dialogue_system.user_persona) > 10:  #ペルソナの数調整
                            dialogue_system.user_persona = dialogue_system.user_persona[len(dialogue_system.user_persona)-10:]
            
        #print(dialogue_system.system_response)
        system_persona_tmp = dialogue_system.extract_persona(dialogue_system.system_response)[1]
        if not "[None]" in system_persona_tmp:
            dialogue_system.system_persona += system_persona_tmp.split("\n")
            if len(dialogue_system.user_persona) > 10:
                dialogue_system.system_persona = dialogue_system.system_persona[len(dialogue_system.system_persona)-10:]


        #dialogue_system.dialogue_history.append({"role": "system", "content": response})

        """
        #デバッグ用
        print(dialogue_system.dialogue_history)     #対話履歴の出力    
        print(dialogue_system.system_persona)       #システムのペルソナ出力
        print(dialogue_system.user_persona)
        """
        


if __name__ == "__main__":
    main()