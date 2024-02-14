# プログラムの実行方法
## 実行環境
OS: Windows11 (Ubuntuでも可)
Python: 3.10.11
PyTorch 2.2.0+cu121
CUDA: 12.1
CUDNN: 8.9.7

した二つのパッケージはなくても動くが、動作は重くなる。

##環境構築
まずPythonの仮想環境を構築する。適当なディレクトリに移動して以下のコマンドを実行
```
python -m venv [仮想環境の名前]
```

すると仮想環境のディレクトリができるのでそこに移動
```
cd [仮想環境のディレクトリまでのパス]
```

Windows(Powershell)の場合は事前に以下のコマンドを実行してスクリプトの実行を許可
```
Set-ExecutionPolicy RemoteSighed -Scope CurrentUser -Force
```

作成した仮想ディレクトリ内のactivateを実行
```
[仮想環境までのパス]\Scripts\activate
#Ubuntuならsourceで実行
source [仮想環境までのパス]/bin/activate
```

仮想環境に入れたら、requirements.txtに必要なパッケージをまとめてあるので、それを基にインストール
```
pip install -r requirements.txt
```

この次にPyTorchをインストールするのだが、CUDAのバージョンに注意してインストールする。
[公式サイト](https://pytorch.org/)を参考にするとよい。
GPUを使って動かしたいと思っていてCUDA、CUDNNをインストールしていない場合は、まずその二つをインストールする。

インストール出来たら**DialogueSystem/DialogueSystem**のOpenAI APIキーを自分のものに差し替える。
最後に`python main.py`で対話システムを動かせる。
コマンドを使ってモデルを変えたりできる。主要なコマンドは以下の通り。

```
--langmodel [gpt3.5-turbo, gpt-4.0] #言語モデルの切り替え(デフォはgpt-3.5)
--whispermodel[tiny,base, small,medium,large,large-v2,large-v3] #音声認識モデルの切り替え(デフォはbase)
--device [cpu, cuda, mps] #cpuかgpuか.pytorchがcudaに対応していればcudaになる.それ以外はcpu.mpsはmacのgpu
```

その他のコマンドはmain.pyを参照。対話終了のトリガーは「さようなら」。これで対話が終了する。