# NDL Classical Books OCR-Lite desktop application
This explains how to use NDL Classical Books OCR-Lite.

## How to start
Extract the compressed file corresponding to your OS, and then double-click ndlkotenocr_lite to run it.
You may get a security warning.
In the case of Windows 10, select “Windows protected your PC” > “More info” > “Run”.
In the case of macOS,
https://zenn.dev/nakamura196/articles/c62a465537ff20
.
Please note that the first time you start the program it may take up to a minute to start. Please wait.

## How to use
<img src=“../resource/control.jpg” width=“600”>
① Specify the file or directory you want to process with OCR.*** From v1.1.1, it is possible to process PDF files in addition to images (jpg, png, tiff, bmp, jp2).***
② Specify the directory where the output will be saved.
③ Click the “OCR” button.
④ If you specified a directory, you can display a preview of the processing results from the images that have been processed with OCR. If you specified a file, the processing results for that file will be displayed.

## How to build the application yourself (information for developers)
This application uses [Flet (external site)] (https://flet.dev/).
You will need to install Flutter-SDK beforehand for any OS. We will omit the explanation of installing the dependencies.

### For Windows
https://flet.dev/docs/publish/windows/
Please also refer to.
```
# (Use the command prompt and run in the same directory as ndlkotenocr-lite-gui)
python3 -m venv ocrenv
.\ocrenv\Scripts\activate
pip install flet==0.24.1
xcopy ..\src .\src
flet build windows
```

### Mac
```
# (Run in the same directory as ndlkotenocr-lite-gui)
python3 -m venv ocrenv
source ./ocrenv/bin/activate
pip install flet==0.24.1
cp -r ../src .
flet build macos
```

### For Linux
```
#(Run in the same directory as ndlkotenocr-lite-gui)
python3 -m venv ocrenv
source ./ocrenv/bin/activate
pip install flet==0.24.1
cp -r ../src .
flet build linux
```

--------------------------------------------
# NDL古典籍OCR-Liteのデスクトップアプリケーション

NDL古典籍OCR-Liteの使い方を説明します。

## 起動方法
お使いのOSに対応する圧縮ファイルを展開し、ndlkotenocr_liteをダブルクリック等で実行してください。

なお、セキュリティに関する警告が出ることがあります。

Windows 10の場合は、「WindowsによってPCが保護されました」→「詳細情報」→「実行」を選んでください。

macOSの場合は、

https://zenn.dev/nakamura196/articles/c62a465537ff20

の手順に従ってください。

なお、初回の起動には1分程度時間を要することがあります。お待ちください。

## 操作方法

<img src="../resource/control.jpg" width="600">

①OCR処理をかけたいファイルまたはディレクトリを指定します。***v1.1.1からは画像（jpg,png,tiff,bmp,jp2）に加えてPDFファイルに対する処理が可能です。***

②出力先となるディレクトリを指定します。

③「OCR」ボタンを押します。

④ ディレクトリを指定した場合、OCR処理が完了した画像から処理結果のプレビューを表示できます。ファイルを指定した場合には当該ファイルの処理結果が表示されます。

## 自分でアプリケーションをビルドする場合の方法（開発者向け情報）
本アプリケーションは[Flet（外部サイト）](https://flet.dev/)を利用します。

いずれのOSの場合にも事前にFlutter-SDKの導入が必要です。依存関係のインストールに関する説明は省略します。

### Windowsの場合
https://flet.dev/docs/publish/windows/
も参照してください。
```
#(コマンドプロンプトを利用、ndlkotenocr-lite-guiと同階層で実行する)
python3 -m venv ocrenv
.\ocrenv\Scripts\activate
pip install flet==0.24.1
xcopy ..\src .\src
flet build windows
```

### Macの場合

```
#(ndlkotenocr-lite-guiと同階層で実行する)
python3 -m venv ocrenv
source ./ocrenv/bin/activate
pip install flet==0.24.1
cp -r ../src .
flet build macos
```

### Linuxの場合
```
#(ndlkotenocr-lite-guiと同階層で実行する)
python3 -m venv ocrenv
source ./ocrenv/bin/activate
pip install flet==0.24.1
cp -r ../src .
flet build linux
```
