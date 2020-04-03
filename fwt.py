" 高速ウェーブレット変換のテスト実装 "
import numpy


def power_of_two(integer):
    """ 入力整数を2の冪数に切り上げる """
    return 2 ** int(numpy.ceil(numpy.log2(integer)))


def generate_scaling_coef(wavelet_coef):
    """ ウェーブレット係数をスケーリング係数に変換 """
    coef_len = len(wavelet_coef)
    sign_array = numpy.array([((-1) ** i) for i in range(coef_len)])
    return sign_array * wavelet_coef[::-1]


def fwt1d(src, scaling_coef, wavelet_coef):
    """ 1次元高速ウェーブレット変換 """
    coef_len = len(scaling_coef)
    src_len = len(src)
    half_src_len = src_len // 2
    decomp_src = numpy.zeros(half_src_len)
    decomp_wav = numpy.zeros(half_src_len)
    # 間引きインデックス
    k = numpy.array(list(range(0, src_len, 2)))
    for n in range(coef_len):
        # インデックスの巡回を考慮
        index = (n + k) % src_len
        decomp_src += scaling_coef[n] * src[index]
        decomp_wav += wavelet_coef[n] * src[index]
    return [decomp_src, decomp_wav]


def ifwt1d(decomp_src, decomp_wav, scaling_coef, wavelet_coef):
    """ 1次元高速ウェーブレット逆変換 """
    half_coef_len = len(scaling_coef) // 2
    decomp_src_len = len(decomp_src)
    src = numpy.zeros(2 * decomp_src_len)
    n = numpy.array(list(range(decomp_src_len)))
    n_even = 2 * n
    n_odd = n_even + 1
    for k in range(half_coef_len):
        # インデックスの巡回を考慮
        index = (n - k) % decomp_src_len
        src[n_even] += scaling_coef[2 * k] * decomp_src[index] \
            + wavelet_coef[2 * k] * decomp_wav[index]
        src[n_odd] += scaling_coef[2 * k + 1] * decomp_src[index] \
            + wavelet_coef[2 * k + 1] * decomp_wav[index]
    return src


def fwt2d(src2d, scaling_coef, wavelet_coef):
    """ 2次元高速ウェーブレット変換 """
    src_len = src2d.shape[0]
    half_src_len = src_len // 2
    src2d_ll = numpy.zeros((half_src_len, half_src_len))
    src2d_hl = numpy.zeros((half_src_len, half_src_len))
    src2d_lh = numpy.zeros((half_src_len, half_src_len))
    src2d_hh = numpy.zeros((half_src_len, half_src_len))
    src2d_l = numpy.zeros((src_len, half_src_len))
    src2d_h = numpy.zeros((src_len, half_src_len))
    # src2dを低域（左）と高域（右）に分解
    for j in range(src_len):
        sl, sh = fwt1d(src2d[j, :], scaling_coef, wavelet_coef)
        src2d_l[j, :] = sl.T
        src2d_h[j, :] = sh.T
    # src2d_l, src2d_hを更に左上(ll)、左下(hl)、右上(lh)、右下(hh)に分割
    for j in range(half_src_len):
        sl, sh = fwt1d(src2d_l[:, j], scaling_coef, wavelet_coef)
        src2d_ll[:, j] = sl
        src2d_hl[:, j] = sh
        sl, sh = fwt1d(src2d_h[:, j], scaling_coef, wavelet_coef)
        src2d_lh[:, j] = sl
        src2d_hh[:, j] = sh
    return [src2d_ll, src2d_lh, src2d_hl, src2d_hh]


def ifwt2d(src2d_ll, src2d_lh, src2d_hl, src2d_hh, scaling_coef, wavelet_coef):
    """ 2次元高速ウェーブレット逆変換 """
    src_len = src2d_ll.shape[0]
    twice_src_len = 2 * src_len
    src2d = numpy.zeros((twice_src_len, twice_src_len))
    src2d_l = numpy.zeros((twice_src_len, src_len))
    src2d_h = numpy.zeros((twice_src_len, src_len))
    # 左上(ll)、左下(hl)、右上(lh)、右下(hh)から左(l)、右(h)に合成
    for j in range(src_len):
        src2d_l[:, j] = ifwt1d(src2d_ll[:, j], src2d_hl[:, j],
                               scaling_coef, wavelet_coef)
        src2d_h[:, j] = ifwt1d(src2d_lh[:, j], src2d_hh[:, j],
                               scaling_coef, wavelet_coef)
    # 左(l)、右(h)から元素材を合成
    for j in range(twice_src_len):
        src2d[j, :] = ifwt1d(src2d_l[j, :], src2d_h[j, :],
                             scaling_coef, wavelet_coef).T
    return src2d


if __name__ == "__main__":
    import sys
    import pickle
    import lzma
    from PIL import Image

    def minmax_scale(vec, maxval):
        minval = numpy.min(vec)
        maxval = numpy.max(vec)
        return maxval * (vec - minval) / (maxval - minval)

    def minmax_clip(vec, minval, maxval):
        vec = numpy.where(vec <= minval, minval, vec)
        vec = numpy.where(vec >= maxval, maxval, vec)
        return vec

    # 画像読み込み
    raw_picture = numpy.asarray(Image.open(sys.argv[1])).astype(int)
    raw_height = raw_picture.shape[0]
    raw_width = raw_picture.shape[1]

    # サイズを2の冪に切り上げ、配列に画像データをロード
    # 余白は0埋め
    extended_width = power_of_two(max(raw_width, raw_height))
    extended = numpy.zeros(shape=(extended_width, extended_width), dtype=int)
    extended[0:raw_height, 0:raw_width] = raw_picture[:, :]
    # 分割結果の画像
    octaveimg = numpy.zeros(shape=(extended_width, extended_width), dtype=int)
    octavecoef = numpy.zeros(shape=(extended_width, extended_width), dtype=int)

    # ウェーブレット係数とスケーリング係数
    # wav_coef = [1, 1]                           # Haar
    # wav_coef /= numpy.linalg.norm(wav_coef)     # 正規化（ドベシィ(1)と一致）
    wav_coef = [0.482962913145, 0.836516303738,
                0.224143868042, -0.129409522551]  # ドベシィ(2)
    # wav_coef = [0.332670552950, 0.806891509311, 0.459877502118,
    #            -0.135011020010, -0.085441273882, 0.035226291882]  # ドベシィ(3)
    # wav_coef = [0.230377813309, 0.714846570553, 0.630880767930,
    #           -0.027983769417, -0.187034811719, 0.030841381836,
    #           0.032883011667, -0.010597401785]  # ドベシィ(4)
    scal_coef = generate_scaling_coef(wav_coef)

    # 解像度の最大深度
    max_dim = int(numpy.log2(extended_width)) - 1  # 最大解像度
    # 多重解像度分析
    octave_list = [extended]
    for dim in range(max_dim):
        # 先頭要素にスケーリング係数成分（左上分割成分）が入っている
        ll_picture = octave_list.pop(0)
        # スケーリング係数成分を4分割
        decomp = fwt2d(ll_picture, wav_coef, scal_coef)
        # 分割結果を画像にセット
        width = decomp[0].shape[0]
        octaveimg[0:width, 0:width] = minmax_scale(decomp[0], 255)
        octaveimg[width:2*width, 0:width] = minmax_scale(decomp[1], 255)
        octaveimg[0:width, width:2*width] = minmax_scale(decomp[2], 255)
        octaveimg[width:2*width, width:2*width] = minmax_scale(decomp[3], 255)
        octavecoef[0:width, 0:width] = decomp[0]
        octavecoef[width:2*width, 0:width] = decomp[1]
        octavecoef[0:width, width:2*width] = decomp[2]
        octavecoef[width:2*width, width:2*width] = decomp[3]
        # octave_listの先頭に結果を挿入
        decomp.extend(octave_list)
        # リスト先頭はoctave_listに再設定
        octave_list = decomp
    # 多重解像度解析の結果を保存
    Image.fromarray(numpy.uint8(octaveimg)).save("test_octave.png")

    # 閾値の計算
    sorted_coef \
        = numpy.sort(numpy.abs(numpy.ndarray.flatten(octavecoef)))[::-1]
    threshould = sorted_coef[len(sorted_coef) // 20]

    # 閾値以下の係数を0に（ハードスレッショルド） 整数に丸め込み
    for i, oct_array in enumerate(octave_list):
        octave_list[i] = numpy.where(numpy.abs(oct_array) <= threshould, 0, oct_array)
        octave_list[i] = numpy.round(octave_list[i]).astype(numpy.int32)

    # LZMAで圧縮してみる
    data = [raw_height, raw_width, octave_list]
    lzma_comp = lzma.LZMACompressor()
    with open("test_compress.xz", "wb") as fout:
        fout.write(lzma_comp.compress(pickle.dumps(data)))
        fout.write(lzma_comp.flush())

    # LZMAを展開
    lzma_decomp = lzma.LZMADecompressor()
    with open("test_compress.xz", "rb") as fin:
        raw_height, raw_width, octave_list = pickle.loads(lzma.decompress(fin.read()))
        
    # 再構成
    while len(octave_list) > 1:
        # 先頭の解析結果から再構成
        composed_img = ifwt2d(octave_list[0], octave_list[1],
                              octave_list[2], octave_list[3], wav_coef, scal_coef)
        # リスト先頭に構成中のデータcomposed_imgを再設定
        octave_list = octave_list[3:]
        octave_list[0] = composed_img
    composed_img = octave_list[0]

    # 画像の書き出し
    # Image.fromarray(numpy.uint8(octave_list[0])).save("test_composed.png")
    # Image.fromarray(numpy.uint8(minmax_clip(octave_list[0], 0, 255))).save("test_composed.png")
    Image.fromarray(numpy.uint8(minmax_clip(composed_img[0:raw_height, 0:raw_width], 0, 255))).save("test_composed.png")
    # Image.fromarray(numpy.uint8(minmax_scale(octave_list[0], 255))).save("test_composed.png")

    sys.exit()
