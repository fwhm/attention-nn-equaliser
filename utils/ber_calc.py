# BER calculation for QAM symbol vectors
# Use QAM_BER_gray function to calculate BER
import numpy as np
from scipy import special
import time
# for JAX
# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def gray_alphabet(bm):

    gseq = np.empty((2 ** bm, bm), dtype=int)
    for i in range(2 ** bm):
        buf = i ^ (i >> 1)
        buf = np.asarray([int(x) for x in bin(buf)[2:]])
        gseq[i, :] = np.append(np.zeros(bm - buf.size, dtype=int), buf)
    return gseq


def gray_qam_bit_abc(m):

    bm = int(m / 2)
    gseq = gray_alphabet(bm)
    gabc = np.concatenate((np.tile(gseq, reps=(2 ** bm, 1)), np.repeat(gseq, repeats=2 ** bm, axis=0)), axis=1)
    return gabc


def gray_qam_sym_abc(m, norm=True):

    ms = int(np.sqrt(2 ** m))
    abc_side = np.arange(0, ms) * 2 - (ms - 1)
    QAM_abc = np.tile(abc_side, reps=ms) + 1j * np.repeat(np.flip(abc_side, axis=0), repeats=ms, axis=0)
    if norm:
        QAM_abc = QAM_abc / np.std(QAM_abc)

    return QAM_abc


def hard_slice(QAMsyms, m, norm=True):

    alphabet = gray_qam_sym_abc(m, norm)
    # sym_indices = list(map(lambda sym: np.argmin(np.abs(sym - alphabet)), QAMsyms))
    sym_indices = [*map(lambda sym: np.argmin(np.abs(sym - alphabet)), QAMsyms)]  # might be faster
    return alphabet[sym_indices], sym_indices


def qam2gray_bits(QAMsyms, QAM_order, norm=True):
    # Converts vector QAM complex-valued symbols to the Gray coded bits
    # QAMsyms - QAM symbol vector to convert
    # QAM_order - order of the QAM target alphabet (e.g. 16 for 16QAM)
    # norm - whether the targer QAM alphabet has unitary power

    m = np.log2(QAM_order)  # Number of bits per QAM symbol

    # Popular error tracking
    if np.mod(m, 1.) != 0.:
        raise ValueError('Given QAM order should be some power of 2.')
    if np.mod(m, 2.) != 0.:
        raise ValueError('Non-square constellations are not supported (e.g. 32QAM, 128QAM)')
    if QAMsyms.ndim != 1:
        raise ValueError('Input array of QAM symbols must be an array')

    m = int(m)  # Convert bit number to integer after checking its value
    QAM_indices = hard_slice(QAMsyms, m, norm)[1]  # Hard slice the input QAM sequence and return its
    bit_alphabet = gray_qam_bit_abc(m)  # Bit patterns, corresponding to every symbol from QAM alphabet
    bit_seq = np.concatenate(tuple((bit_alphabet[QAM_ind] for QAM_ind in QAM_indices)), axis=0)
    return bit_seq


def qam_ber_gray(QAMsyms_chk, QAMsyms_ref, QAM_order, norm=True):
    # Calculates BER between the two QAM symbol vectors in input data
    # QAMsyms - QAM symbol vector to convert
    # QAM_order - order of the QAM target alphabet (e.g. 16 for 16QAM)
    # norm - whether the targer QAM alphabet has unitary power

    bits_chk = qam2gray_bits(QAMsyms_chk, QAM_order, norm)
    bits_ref = qam2gray_bits(QAMsyms_ref, QAM_order, norm)
    BER = np.mean(np.logical_xor(bits_ref, bits_chk))
    return BER

# Construct NMSE estimator
def nmse(x_in, x_ref):
    return 20. * np.log10(np.linalg.norm(x_in - x_ref) / np.linalg.norm(x_ref))

def ber2q(BER):
    return 20 * np.log10(np.sqrt(2) * special.erfcinv(2 * BER))