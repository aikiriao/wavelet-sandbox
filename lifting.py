" Liftingのテスト実装 "
import numpy


def power_of_two(integer):
    """ 入力整数を2の冪数に切り上げる """
    return 2 ** int(numpy.ceil(numpy.log2(integer)))


def div_ceil(vec, divider):
    """ vecをdividerで除算して小数部を切り上げた結果を返す """
    return -((-vec) // divider)


def primal_lifting_haar(decomp_src, decomp_wav):
    """ Haar基底によるPrimal Lifting """
    decomp_wav -= decomp_src
    decomp_src += div_ceil(decomp_wav, 2)
    return [decomp_src, decomp_wav]


def dual_lifting_haar(decomp_src, decomp_wav):
    """ Haar基底によるDual Lifting """
    decomp_src -= div_ceil(decomp_wav, 2)
    decomp_wav += decomp_src
    return [decomp_src, decomp_wav]


def primal_lifting_cdf2x(decomp_src, decomp_wav):
    """ CDF2/X基底によるPrimal Lifting """
    decomp_wav -= div_ceil((decomp_src + numpy.append(decomp_src[1:], decomp_src[-1])), 2)
    decomp_src += div_ceil((numpy.append(decomp_src[0], decomp_src[:-1]) + decomp_wav), 4)
    # decomp_wav -= div_ceil((decomp_src + numpy.roll(decomp_src, -1)), 2)
    # decomp_src += div_ceil((numpy.roll(decomp_wav, 1) + decomp_wav), 4)
    return [decomp_src, decomp_wav]


def dual_lifting_cdf2x(decomp_src, decomp_wav):
    """ CDF2/X基底によるDual Lifting """
    decomp_src -= div_ceil((numpy.append(decomp_src[0], decomp_src[:-1]) + decomp_wav), 4)
    decomp_wav += div_ceil((decomp_src + numpy.append(decomp_src[1:], decomp_src[-1])), 2)
    # decomp_src -= div_ceil((numpy.roll(decomp_wav, 1) + decomp_wav), 4)
    # decomp_wav += div_ceil((decomp_src + numpy.roll(decomp_src, -1)), 2)
    return [decomp_src, decomp_wav]


def fwt2d_lifting_haar(src2d):
    """ Haar基底による2次元ウェーブレット変換 """
    src_len = src2d.shape[0]
    half_src_len = src_len // 2
    src2d_ll = numpy.zeros((half_src_len, half_src_len), dtype=int)
    src2d_hl = numpy.zeros((half_src_len, half_src_len), dtype=int)
    src2d_lh = numpy.zeros((half_src_len, half_src_len), dtype=int)
    src2d_hh = numpy.zeros((half_src_len, half_src_len), dtype=int)
    src2d_l = numpy.zeros((src_len, half_src_len), dtype=int)
    src2d_h = numpy.zeros((src_len, half_src_len), dtype=int)
    # スプリットしてPrimal Lifting
    # 画像を低域（左）と高域（右）に分解
    for j in range(src_len):
        sl, sh = primal_lifting_haar(src2d[j, 0::2], src2d[j, 1::2])
        src2d_l[j, :] = sl.T
        src2d_h[j, :] = sh.T
    # src2d_l, src2d_hを更に左上(ll)、左下(hl)、右上(lh)、右下(hh)に分割
    for j in range(half_src_len):
        src2d_ll[:, j], src2d_hl[:, j] = primal_lifting_haar(src2d_l[0::2, j], src2d_l[1::2, j])
        src2d_lh[:, j], src2d_hh[:, j] = primal_lifting_haar(src2d_h[0::2, j], src2d_h[1::2, j])
    # 正規化
    # src2d_ll *= 2
    # src2d_hl *= 2
    # src2d_lh *= 2
    # src2d_hh *= 2
    return [src2d_ll, src2d_lh, src2d_hl, src2d_hh]


def ifwt2d_lifting_haar(src2d_ll, src2d_lh, src2d_hl, src2d_hh):
    """ Haar基底による2次元ウェーブレット逆変換 """
    src_len = src2d_ll.shape[0]
    twice_src_len = 2 * src_len
    src2d = numpy.zeros((twice_src_len, twice_src_len), dtype=int)
    src2d_l = numpy.zeros((twice_src_len, src_len), dtype=int)
    src2d_h = numpy.zeros((twice_src_len, src_len), dtype=int)
    # 正規化を解く
    # src2d_ll //= 2
    # src2d_hl //= 2
    # src2d_lh //= 2
    # src2d_hh //= 2
    # 左上(ll)、左下(hl)、右上(lh)、右下(hh)から左(l)、右(h)に合成
    for j in range(src_len):
        src2d_l[0::2, j], src2d_l[1::2, j] = dual_lifting_haar(src2d_ll[:, j], src2d_hl[:, j])
        src2d_h[0::2, j], src2d_h[1::2, j] = dual_lifting_haar(src2d_lh[:, j], src2d_hh[:, j])
    # 左(l)、右(h)から元素材を合成
    for j in range(twice_src_len):
        sl, sh = dual_lifting_haar(src2d_l[j, :], src2d_h[j, :])
        src2d[j, 0::2] = sl.T
        src2d[j, 1::2] = sh.T
    return src2d


if __name__ == "__main__":
    import sys
    import lzma
    import pickle
    from PIL import Image

    def minmax_scale(vec, maxval):
        minval = numpy.min(vec)
        maxval = numpy.max(vec)
        return maxval * (vec - minval) / (maxval - minval)

    # 画像読み込み
    raw_picture = numpy.asarray(Image.open(sys.argv[1]))
    raw_height = raw_picture.shape[0]
    raw_width = raw_picture.shape[1]

    # サイズを2の冪に切り上げ、配列に画像データをロード
    # 余白は0埋め
    extended_width = power_of_two(max(raw_width, raw_height))
    extended = numpy.zeros(shape=(extended_width, extended_width), dtype=int)
    extended[0:raw_height, 0:raw_width] = raw_picture[:, :]
    # 分割結果の画像
    octaveimg = numpy.zeros(shape=(extended_width, extended_width), dtype=int)

    # 多重解像度解析
    # max_dim = int(numpy.log2(extended_width)) - 1
    max_dim = 2
    octave_list = [extended.copy()]
    for dim in range(max_dim):
        # 先頭要素（スケーリング係数成分）の取り出し
        src2d = octave_list.pop(0)
        # スケーリング係数成分を4分割
        decomp = fwt2d_lifting_haar(src2d)
        # 分割結果を画像にセット
        width = decomp[0].shape[0]
        octaveimg[0:width, 0:width] = minmax_scale(decomp[0], 255)
        octaveimg[width:2*width, 0:width] = minmax_scale(decomp[1], 255)
        octaveimg[0:width, width:2*width] = minmax_scale(decomp[2], 255)
        octaveimg[width:2*width, width:2*width] = minmax_scale(decomp[3], 255)
        # リスト先頭に結果を挿入
        decomp.extend(octave_list)
        # リスト先頭はoctave_listに再設定
        octave_list = decomp

    # 多重解像度解析の結果を保存
    Image.fromarray(numpy.uint8(octaveimg)).save("test_octave.png")

    # 再構成
    while len(octave_list) > 1:
        composed_img = ifwt2d_lifting_haar(octave_list[0], octave_list[1],
                                           octave_list[2], octave_list[3])
        # リスト先頭に構成中のデータを再設定
        octave_list = octave_list[3:]
        octave_list[0] = composed_img
    composed_img = octave_list[0]

    # 画像の書き出し
    Image.fromarray(numpy.uint8(composed_img[0:raw_height, 0:raw_width])).save("test_composed.pgm")

    sys.exit()

