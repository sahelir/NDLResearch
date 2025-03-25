# Repository for the NDL Classical Books OCR-Lite application
This is a repository that provides applications for performing text conversion using the NDL Classical Books OCR-Lite.
NDL Classical Books OCR-Lite is an OCR that creates text data from digitized images of classical book materials such as Japanese old books from before the Edo period and Chinese books from before the Qing dynasty.
It is characterized by OCR processing that does not require a GPU, and can be executed quickly on general home computers such as laptop computers and OS environments.
It has been confirmed to work on Windows (Windows 10), Intel Mac (macOS Sequoia), and Linux (Ubuntu 22.04) environments.
There are some documents that it is better at and some that it is not, but when compared to [NDL Classical Books OCR ver.3] (https://github.com/ndl-lab/ndlkotenocr_cli), it is known that the text conversion accuracy is about 2% lower. In the future, we will work to improve the accuracy to a level equivalent to or higher than NDL Classical Books OCR.
This program was developed independently by utilizing the knowledge gained through the research activities of the [NDL Laboratory] (https://lab.ndl.go.jp) and data resources that have been constructed, accumulated, and published in the field of humanities informatics, including the [Transcription Data for Everyone (external site)] (https://github.com/yuta1984/honkoku-data).
For details of the data sets used in the development and improvement of this program, please also refer to [Experiment in OCR Text Conversion of Classical Books](https://lab.ndl.go.jp/data_set/r4ocr/r4_koten/) and [OCR Learning Data Set (Transcription by Everyone)](https://github.com/ndl-lab/ndl-minhon-ocrdataset).
This program is released by the National Diet Library under the CC BY 4.0 license. For details, please see [LICENCE](./LICENCE). For the license of libraries used when executing this application, please see [LICENCE_DEPENDENCIES](./LICENCE_DEPENDENCEIES).

## Using the desktop application
**When using the desktop application, please place the application in a path that does not contain Japanese (full-width characters). The application may not start if it contains full-width characters.**
Download the file that matches your OS environment (Windows/Mac/Linux) from [releases](https://github.com/ndl-lab/ndlkotenocr-lite/releases).
If you are using it in a Mac environment, please also refer to the following article.
https://zenn.dev/nakamura196/articles/c62a465537ff20
After extracting the zip file, double-click “ndlkotenocr_lite” to run it.
For information on how to use and build the desktop application, please refer to [How to use the desktop application](./ndlkotenocr-lite-gui/README.md).

The following gif animation shows a demo of converting “Fukagawa Kinshiro, et al. ‘Kameya Mannen Urashima Sakae 2-kan’, published in 1783 [Tenmei 3]. https://dl.ndl.go.jp/pid/8929445
“ into text using the Windows desktop application.
<img src=”resource/demo.gif“ width=”600">

## Using from the command line
* Python 3.10 or later is required to operate from the command line.
Preliminary preparation
```
git clone https://github.com/ndl-lab/ndlkotenocr-lite
cd ndlkotenocr-lite
pip install -r requirements.txt
cd src
```
Example 1 (Batch process the images in the directory named “Ryugu Kukai Tamatebako _ 3_9892834_0001” in the same directory, and output the results to a directory named tmpdir.
```
python3 ocr.py --sourcedir Ryugu Kugai Tamatebako _ 3巻_9892834_0001 --output tmpdir 
```
Example 2 (Processes the image named “digidepo_1287221_00000002.jpg” in the same directory and outputs the results to a directory named tmpdir.
```
python3 ocr.py --sourceimg digidepo_1287221_00000002.jpg --output tmpdir 
```
### Parameter descriptions
#### `--sourcedir` option
Specify the directory containing the images you want to process, using either an absolute or relative path. The program will process files with the extensions “jpg (jpeg is also acceptable)”, “png”, “tiff (tif is also acceptable)”, “jp2”, and “bmp” in the directory in sequence.
#### `--sourceimg` option
Specify the image you want to process directly using an absolute or relative path. It is possible to process files with the extensions “jpg (jpeg is also possible)”, “png”, “tiff (tif is also possible)”, “jp2”, and “bmp”.
#### `--output` option
Specify the output directory to save the OCR results to using an absolute or relative path.
#### `--viz` option
By specifying `--viz True`, an image with the character recognition areas displayed in blue frames will be output to the output directory.
#### `--device` option (beta)
Only on servers with compatible GPUs and on environments where onnxruntime-gpu is installed, specifying `--device cuda` will switch to processing using the GPU.

## Example of OCR results
OMIT FOR BREVITY

## About model re-learning and customization (information for developers)
Please see [Learning and model conversion procedure](/train/README.md).

## About technical information (information for developers)
NDL OCR-Lite for Historical Japanese Books is realized by combining three functions (modules): “layout recognition”, “character string recognition”, and “reading order sorting”.
RTMDet[1] is used for layout recognition, and PARSeq[2] is used for character string recognition. The same module as that used in [NDL Classical Books OCR ver.3] (https://github.com/ndl-lab/ndlkotenocr_cli), which is publicly available from our library, is used for reading order sorting.
[1]Chengqi Lyu, Wenwei Zhang, Haian Huang, Yue Zhou, Yudong Wang, Yanyi Liu, Shilong Zhang, Kai Chen. Rtmdet: An empirical study of designing real-time object detectors. arXiv preprint arXiv:2212.07784, 2022.(https://arxiv.org/abs/2212.07784)
[2]Darwin Bautista, Rowel Atienza. Scene text recognition with permuted autoregressive sequence models. arXiv:2212.06966, 2022. (https://arxiv.org/abs/2207.06966)
Both the layout recognition and character string recognition machine learning models were trained using pytorch as the framework, and then converted to ONNX format for use. For details, please see [Training and Model Conversion Procedure](/train/README.md).

For more detailed information on the development background and technical considerations, please see the following paper.
Aoike, Toru. Development of “NDL Classical Books OCR-Lite”, a lightweight OCR that runs quickly in a CPU environment. Proceedings of the Symposium on Humanities and Computers : Jinmonkon 2024 (IPSJ symposium series ; vol. 2024 no. 1). Information Processing Society of Japan, 2024.12 [External link](https://ipsj.ixsq.nii.ac.jp/records/241527)
[Poster](./resource/ndl_jinmonkon2024.pdf)

------------------------------------------
# NDL古典籍OCR-Liteアプリケーションのリポジトリ

NDL古典籍OCR-Liteを利用してテキスト化を実行するためのアプリケーションを提供するリポジトリです。

NDL古典籍OCR-Liteは、江戸期以前の和古書、清代以前の漢籍といった古典籍資料のデジタル化画像からテキストデータを作成するOCRです。

GPUを必要としないOCR処理に特徴があり、ノートパソコン等の一般的な家庭用コンピュータやOS環境において高速に実行可能です。

Windows(Windows 10)、Intel Mac(macOS Sequoia)及びLinux(Ubuntu 22.04)環境において動作確認しています。

資料により得意・不得意がありますが[NDL古典籍OCR ver.3](https://github.com/ndl-lab/ndlkotenocr_cli)と比較するとテキスト化精度が2%程度低下することが分かっています。今後、精度についてもNDL古典籍OCRと同等以上の水準を目指して改善を図ります。

本プログラムは[NDLラボ](https://lab.ndl.go.jp)におけるこれまでの調査研究活動によって得られた知見等や[みんなで翻刻データ（外部サイト）](https://github.com/yuta1984/honkoku-data)をはじめとする人文情報学分野において構築・蓄積・公開されてきたデータ資源を活用することで独自に開発したものです。

本プログラムを開発・改善するに当たって利用したデータセット等の詳細については、[古典籍資料のOCRテキスト化実験](https://lab.ndl.go.jp/data_set/r4ocr/r4_koten/)及び[OCR学習用データセット（みんなで翻刻）](https://github.com/ndl-lab/ndl-minhon-ocrdataset)も参照してください。

本プログラムは、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については[LICENCE](./LICENCE)をご覧ください。なお、本アプリケーションの実行時に利用するライブラリ等のライセンスについては[LICENCE_DEPENDENCIES](./LICENCE_DEPENDENCEIES)をご覧ください。


## デスクトップアプリケーションによる利用

**デスクトップアプリケーションを利用する際には、日本語（全角文字）を含まないパスにアプリケーションを配置してください。全角文字を含む場合に起動しないことがあります。**

[releases](https://github.com/ndl-lab/ndlkotenocr-lite/releases)からお使いのOS環境（Windows/Mac/Linux）に合ったファイルをダウンロードしてください。

Mac環境における利用の場合には、次の記事も参考にしてください。

https://zenn.dev/nakamura196/articles/c62a465537ff20

zipファイルを展開後、「ndlkotenocr_lite」をダブルクリックする等で実行してください。

デスクトップアプリケーションの操作方法及びビルド方法については[デスクトップアプリケーションの利用方法](./ndlkotenocr-lite-gui/README.md)を参照してください。


次のgifアニメーションは、"深川錦鱗 作 ほか『亀屋万年浦島栄 2巻』,刊,天明3 [1783]. https://dl.ndl.go.jp/pid/8929445
"をWindows版デスクトップアプリケーションによってテキスト化するデモを示しています。

<img src="resource/demo.gif" width="600">

## コマンドラインからの利用
※コマンドラインから操作を行うにはPython 3.10以上が必要です。

事前準備
```
git clone https://github.com/ndl-lab/ndlkotenocr-lite
cd ndlkotenocr-lite
pip install -r requirements.txt
cd src
```
実行例1.（同階層にある「竜宮苦界玉手箱 _ 3巻_9892834_0001」という名称のディレクトリ内の画像を一括処理し、tmpdirという名称のディレクトリに結果を出力する。）
```
python3 ocr.py --sourcedir 竜宮苦界玉手箱 _ 3巻_9892834_0001 --output tmpdir 
```

実行例2.（同階層にある「digidepo_1287221_00000002.jpg」という名称の画像を処理し、tmpdirという名称のディレクトリに結果を出力する。）
```
python3 ocr.py --sourceimg digidepo_1287221_00000002.jpg --output tmpdir 
```

### パラメータの説明

#### `--sourcedir`オプション
処理したい画像の含まれるディレクトリを絶対パスまたは相対パスで指定する。ディレクトリ内の"jpg（jpegも可）"、"png"、"tiff（tifも可）"、"jp2"及び"bmp"の拡張子のファイルを順次処理する。

#### `--sourceimg`オプション
処理したい画像を絶対パスまたは相対パスで直接指定する。"jpg（jpegも可）"、"png"、"tiff（tifも可）"、"jp2"及び"bmp"の拡張子のファイルを処理することが可能。

#### `--output`オプション
OCR結果を保存する出力先ディレクトリを相対パスまたは絶対パスで指定する。

#### `--viz`オプション
`--viz True`を指定することで、文字認識箇所を青枠で表示した画像を出力先ディレクトリに出力する。

#### `--device`オプション（ベータ）
対応GPUを搭載したサーバかつonnxruntime-gpuがインストールされている環境に限り、`--device cuda`を指定することでGPUを利用した処理に切り替える。


## OCR結果の例


|資料画像（文字認識箇所を青枠で表示）|OCR結果（誤認識を含む）|
|---|---|
|<img src="./resource/1287221_0002.jpg" width="400"><br>『竹取物語』上,江戸前期. https://dl.ndl.go.jp/pid/1287221/1/2|いまはむかしたけとりのおきなといふ<br>ものありけり野山にましりてたけ<br>をとりつゝよろつの事につかひけり<br>名をはさるきのみやつことなんいひ<br>ける其竹の中にもとひかる竹なん<br>一すちありけりあやしかりてよりて<br>見るにつゝの中ひかりたりそれをみ<br>れは三すんはかりなる人いとくつくしう<br>てゐたりおきないふやうわれ朝こと夕<br>ことに見るたけの中におはするにて<br>しりぬ子になり給ふへき人なめり<br>とて手にうち入て家へもちてきぬ<br>めの女にあつけてやしなはすうつ<br>くしき事かきりなしいとおさなけ<br>れははこに入てやしなふ竹とりのお<br>さな竹とるに此子を見つけてのちに|
|<img src="./resource/10301438_0017.jpg" width="400"><br> 曲亭馬琴 作 ほか『人間万事賽翁馬 3巻』,鶴喜,寛政12. https://dl.ndl.go.jp/pid/10301438/1/17|馬九郎わつとなくこんと<br>ともにきこゆるゑんじ<br>のかねあたりを<br>みればコはいかに<br>さいわうじのやか<br>たとみしはもと<br>やすらひたるのはら<br>にておや子ゆめさめ<br>おどろけば馬もの<br>そのまゝつな<br>がれてたい<br>くつそうな<br>るはないばい<br>さてはゆめ<br>かとおや<br>子がうれ<br>しさ<br>むねは<br>ぜんたり<br>ときに<br>ぎや<br>くわんぜ<br>をんから<br>やくと<br>ひかり<br>近はな<br>ちむ<br>りやうにつかしぎの<br>方便をといて<br>馬介郎おや子<br>をみちびき<br>たもふ<br>そのときくわんおんつけたまわくどふだ馬五郎<br>おそれるからくはくのたねさいわいはわざわい<br>のもとすこしのさいわいわいあればまたすこ<br>のわざわひあり大イなるさいわいにあへばかなら<br>ず大イなるわざわいありそのわざわいはよく<br>しんから心のこまにたづなをゆるしや〳〵<br>鳥のけんくわこうろん人くひ馬のわるだくみ<br>いろくるひのまめごのみそれからしゆらのた<br>いこをうつぼんぶのよくにめくら馬ついにい<br>もへごしとなりてぶい〳〵無為をたのして<br>だしたゞさいわいもなくわざわい<br>もなく無事をにんげんのたからと<br>いふなり積善のいへ余慶あり<br>積悪のいへ余欲あり馬九郎<br>さらばチヤう<br>チヤ〳〵チヤ<br>かきけすごとく<br>うせたもふ<br>ざん〳〵<br>あやまり<br>で入りました<br>〽ありがたや〳〵|
|<img src="./resource/11892692_0004.jpg" width="400"><br> 『論語10卷』一,江戸. https://dl.ndl.go.jp/pid/11892692/1/4|適齊為高昭子家臣以通乎景公公欲封<br>以尼谿之田憂嬰不可公感之孔子遂行<br>及乎魯定公元年壬辰孔子年四十三丙<br>季氏強僭其臣陽虎作乱専政故孔子不<br>仕而退脩詩書礼楽弟子彌衆九年庚子<br>孔子年五十一、公山不伍以費畔季氏召<br>孔子欲往而卒不行。定公以孔子為中都<br>宰一年四方則之遂為司空文為大司冦。<br>十年辛丑相定公会斉侯于夾谷齊人帰<br>魯侵地十二年癸卯使仲由爲季氏宰-<br>三都收其甲兵孟氏不肯隨成団之不克<br>十四年乙巳孔子年五十六攝行相事誅<br>寅正卯與聞国政三月魯国大治［ノ］国大治［ニ］暦<br>玄楽以組之季桓子受之。郊又不致勝組<br>於大夫￣ニ孔子行適衛子路妻兄顔濁<br>郷家適陳過過臣人以為陽陽虎而拘之。既|
|<img src="./resource/9973821_0004.jpg" width="400"><br> 伊藤博文書簡　岩倉具視宛, 明15.8. https://dl.ndl.go.jp/pid/9973821/1/4 |両氏の其主説ハ守鶴ニ傾斜セル者ト翁に候昨日<br> スタインヒ一面銭とも既ニ其説ク所英仏仏獨三ケ国ノ<br>国体及ヒ其国ノ甚阿馬ノ主説トスル所ヲ分別シテ以<br> テ御坐ノ整格ヲ興起セシメ中ニ其概略ヲ申上ハ三国何<br> レモ議政体ナレル其精神大ニ焉ナル者アリ亜人ノ説ク<br> 所ハ政府ナルモノハ桁砕ク国会ニ於テ衆論ノ多数ヲ占ソ<br> タル堂派ノ首領タルモノ政治ヲ施設スル所ト云仏人ハ<br> 政有ハ国会衆議ノ臣僅ナリト云偶人ハ政府タル者ハ<br> 衆議ヲ採ルモ独立行為ノ権アリト云若シ此独立行<br> 為ノ権ナケレハ国会若シ其国費ヲ供給セサル時ハ各<br> ヲ束シテカ国政ヲ放擲セサルウ得ス豈ニ斯ノ如キノ理<br> アランヤ況ンヤ居主ハ立活行双ノ大権ヲ親ヲ掌トリ<br> 君主ノ認可ヲ得スシテ一モ法律ト為ル者ナク君主ノ<br> 許諾ヲ得スシテ一モ施設スルコトナキノ主脳タルニ於テ<br>オヤ由是観之邦国ハ乃チ君主ニシテ君主乃チ邪国ト云<br> モ可ナリ然レトトコト異ナル者アリ立憲君主ノ国ニ在<br>即各宰相<br> 及ヒ百般ノ<br> ノ恨同ナリ<br> テハ立法ノ組織幅蠟院行政ノ組織湯<br> 政治皆ヲ一定ノ組織組津波ニ随テ運用スル是ナリ<br> 大雷如斯ト雖モ之ヲ学問上ノ分易定哉ニ依リ申上候時ハ<br> 勿論片紙ノ所尽ニ無之一又甚長ニ渉リ妙モ無<br> 事ニ坐候故略之申ニ<br> 方栖川完既二何左利二御着中旬頃迄御滞在夫ゟ仏<br> 国へ向チ候発途ニ相成との桶草ゟ一昨日|


## モデルの再学習及びカスタマイズについて（開発者向け情報）
[学習及びモデル変換手順](/train/README.md)をご覧ください。


## 技術情報について（開発者向け情報）

NDL古典籍OCR-Liteは「レイアウト認識」、「文字列認識」、「読み順整序」の3つの機能（モジュール）を組み合わせて実現しています。

レイアウト認識にはRTMDet[1]、文字列認識にはPARSeq[2]をそれぞれ用いており、読み順整序については当館が公開している[NDL古典籍OCR ver.3](https://github.com/ndl-lab/ndlkotenocr_cli)と同様のモジュールを用いています。

[1]Chengqi Lyu, Wenwei Zhang, Haian Huang, Yue Zhou, Yudong Wang, Yanyi Liu, Shilong Zhang, Kai Chen. Rtmdet: An empirical study of designing real-time object detectors. arXiv preprint arXiv:2212.07784, 2022.(https://arxiv.org/abs/2212.07784)

[2]Darwin Bautista, Rowel Atienza. Scene text recognition with permuted autoregressive sequence models. arXiv:2212.06966, 2022. (https://arxiv.org/abs/2207.06966)

レイアウト認識及び文字列認識の機械学習モデルは、いずれもpytorchをフレームワークとした学習を行った後にONNX形式に変換して利用しています。詳しくは[学習及びモデル変換手順](/train/README.md)をご覧ください。


開発背景及び技術検討に関するより詳細な情報については、次の論文をご覧ください。

青池亨. CPU環境で高速に動作する軽量OCR「NDL古典籍OCR-Lite」の開発. 人文科学とコンピュータシンポジウム論文集 : じんもんこん2024 (情報処理学会シンポジウムシリーズ = IPSJ symposium series ; vol. 2024 no. 1). 情報処理学会, 2024.12[外部リンク](https://ipsj.ixsq.nii.ac.jp/records/241527)

[ポスター](./resource/ndl_jinmonkon2024.pdf)
