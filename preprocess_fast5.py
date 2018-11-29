#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
from collections import Counter
from itertools import zip_longest
import os
import shutil


from fairseq.data import indexed_dataset, dictionary
from fairseq.tokenizer import Tokenizer, tokenize_line
from multiprocessing import Pool, Manager, Process


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--source-lang", default=None, metavar="SRC", help="source language"
    )
    parser.add_argument(
        "-t", "--target-lang", default=None, metavar="TARGET", help="target language"
    )
    parser.add_argument(
        "--destdir", metavar="DIR", default="data-bin", help="destination dir"
    )
    parser.add_argument(
        "--thresholdtgt",
        metavar="N",
        default=0,
        type=int,
        help="map words appearing less than threshold times to unknown",
    )
    parser.add_argument(
        "--nwordstgt",
        metavar="N",
        default=-1,
        type=int,
        help="number of target words to retain",
    )
    parser.add_argument("--tgtdict", metavar="FP", help="reuse given target dictionary")

    parser.add_argument(
        "--padding-factor",
        metavar="N",
        default=8,
        type=int,
        help="Pad dictionary size to be multiple of N",
    )
    parser.add_argument(
        "--workers", metavar="N", default=1, type=int, help="number of parallel workers"
    )
    parser.add_argument(
        "--only-source", action="store_true", help="Only process the source language"
    )
    parser.add_argument(
        "--meta", metavar="FP", default=None, help="meta file location"
    )
    parser.add_argument('--chrom', default='Chromosome',
                        help='Chromosome')
    parser.add_argument('--chrom-start', default='0',
                        help='Chromosome start')
    parser.add_argument('--chrom-end', default='1',
                        help='Chromosome end')
    parser.add_argument('--basecall_group', default='RawGenomeCorrected_000',
                        help='Basecall group Nanoraw resquiggle into. Default is Basecall_1D_000')
    parser.add_argument('--basecall_subgroup', default='BaseCalled_template',
                        help='Basecall subgroup Nanoraw resquiggle into. Default is BaseCalled_template')
    parser.add_argument('-l', '--length', default=512, help="Length of the signal segment")
    parser.add_argument('-b', '--batch', default=10000, help="Number of record in one file.")
    parser.add_argument('-n', '--normalization', default='median',
                        help="The method of normalization applied to signal, Median(default):robust median normalization, 'mean': mean normalization, 'None': no normalizaion")
    parser.add_argument('-r', '--replace', default=False, help="Change CPG to M.")

    return parser


def main(args):
    print(args)
    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    group_src, group_tgt = readH5(args,args.destdir,'train')
    # readH5(args,args.destdir,'valid')
    # readH5(args,args.destdir,'test')

    if target:
        if args.tgtdict:
            tgt_dict = dictionary.Dictionary.load(args.tgtdict)
        else:
            assert (
                args.trainpref
            ), "--trainpref must be set if --tgtdict is not specified"
            tgt_dict = build_dictionary(
                group_tgt, args.workers
            )
        tgt_dict.finalize(
            threshold=args.thresholdtgt,
            nwords=args.nwordstgt,
            padding_factor=args.padding_factor,
        )
        tgt_dict.save(dict_path(args.target_lang))

    print("| Wrote preprocessed data to {}".format(args.destdir))

DNA_BASE = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'M':4}
DNA_IDX = ['A', 'C', 'G', 'T','M']

import utils.labelop as labelop
import numpy as np
from statsmodels import robust


def readH5(FLAGS, output_folder, output_prefix, num_workers=1):

    group_h5 = []
    group_src = []
    group_tgt = []

    with open(FLAGS.meta,'r') as file:
        while True:
            line = file.readline()
            if line:
                tmpgroup = line.strip().split()
                if int(tmpgroup[3]) > FLAGS.chrom_start and int(tmpgroup[4]) < FLAGS.chrom_end and tmpgroup[1] == FLAGS.chrom:
                    group_h5.append(tmpgroup[0])

            else:
                break

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    batch_idx = 1
    output_file_input = None
    output_file_label = None
    event = list()
    event_length = list()
    label = list()
    label_length = list()
    success_list = list()
    fail_list = list()
    batch_fast5_idx = 1

    def extract_fast5(input_file_path, bin_h, bin_h_label,mode='DNA'):
        """
        Extract the signal and label from a single fast5 file
        Args:
            input_file_path: path of a fast5 file.
            bin_h: handle of the binary file.
            mode: The signal type dealed with. Default to 'DNA'.
        """
        try:
            (raw_data, raw_label, raw_start, raw_length) = labelop.get_label_raw(input_file_path, FLAGS.basecall_group,
                                                                                 FLAGS.basecall_subgroup)
        except IOError:
            fail_list.append(input_file_path)
            return False
        except:
            fail_list.append(input_file_path)
            return False

        if mode=='rna':
            print(type(raw_data))
            raw_data = raw_data[::-1]
        if FLAGS.normalization == 'mean':
            raw_data = (raw_data - np.median(raw_data)) / np.float(np.std(raw_data))
        elif FLAGS.normalization == 'median':
            raw_data = (raw_data - np.median(raw_data)) / np.float(robust.mad(raw_data))
        pre_start = raw_start[0]
        pre_index = 0
        for index, start in enumerate(raw_start):
            if start - pre_start > FLAGS.length:
                if index - 1 == pre_index:
                    # If a single segment is longer than the maximum singal length, skip it.
                    pre_start = start
                    pre_index = index
                    continue
                event.append(np.pad(raw_data[pre_start:raw_start[index - 1]],
                                    (0, FLAGS.length + pre_start - raw_start[index - 1]), mode='constant'))
                event_length.append(int(raw_start[index - 1] - pre_start))
                label_ind = raw_label['base'][pre_index:(index - 1)]
                temp_label = []
                for ind_x in range(0,len(label_ind)):
                    if FLAGS.replace and label_ind[ind_x] == 'C' and ind_x + 1 < len(label_ind) and label_ind[ind_x + 1] == 'G':
                        temp_label.append(DNA_BASE['M'.decode('UTF-8')])
                    else:
                        temp_label.append(DNA_BASE[label_ind[ind_x].decode('UTF-8')])
                # temp_label = [DNA_BASE[x.decode('UTF-8')] for x in label_ind]
                label.append(
                    np.pad(temp_label, (0, FLAGS.length - index + 1 + pre_index), mode='constant', constant_values=-1))
                label_length.append(index - 1 - pre_index)
                pre_index = index - 1
                pre_start = raw_start[index - 1]
            if raw_start[index] - pre_start > FLAGS.length:
                # Skip a single event segment longer than the required signal length
                pre_index = index
                pre_start = raw_start[index]
        success_list.append(input_file_path)

        while len(event) > FLAGS.batch:
            for index in range(0, FLAGS.batch):
                bin_h.write(event[index].tolist())
                bin_h_label.write(label[index].tolist())
            del event[:FLAGS.batch]
            del event_length[:FLAGS.batch]
            del label[:FLAGS.batch]
            del label_length[:FLAGS.batch]
            return True
        return False

    for file_n in group_h5:
        if file_n.endswith('fast5'):
            if output_file_input is None:
                output_file_input = open(output_folder + os.path.sep + output_prefix +'.input-label.input.'+ str(batch_idx), 'wb+')
                output_file_label = open(output_folder + os.path.sep + output_prefix +'.input-label.label.'+ str(batch_idx), 'wb+')
            output_state = extract_fast5(file_n, output_file_input, output_file_label)
            print("    %d fast5 file done. %d signals read already.\n" % (batch_fast5_idx,len(event)))
            batch_fast5_idx += 1
            if output_state:
                batch_idx += 1
                output_file_input.close()
                output_file_label.close()
                output_file_input = open(output_folder + os.path.sep + output_prefix +'.input-label.input.'+ str(batch_idx), 'wb+')
                output_file_label = open(output_folder + os.path.sep + output_prefix +'.input-label.label.'+ str(batch_idx), 'wb+')
                group_src.append(output_folder + os.path.sep + output_prefix +'.input-label.input.'+ str(batch_idx))
                group_tgt.append(output_folder + os.path.sep + output_prefix +'.input-label.label.'+ str(batch_idx))
                print("%d batch transferred completed.\n" % (batch_idx - 1))


    print("File batch transfer completed, %d batches have been processed\n" % (batch_idx - 1))
    print("%d files scussesfully read, %d files failed.\n" % (len(success_list), len(fail_list)))

    if not output_state:
        output_file_input.close()
        output_file_label.close()
        os.remove(output_folder + os.path.sep + output_prefix +'.input-label.input.' + str(batch_idx))
        os.remove(output_folder + os.path.sep + output_prefix +'.input-label.label.'+ str(batch_idx))
        group_src.remove(output_folder + os.path.sep + output_prefix + '.input-label.input.' + str(batch_idx))
        group_tgt.remove(output_folder + os.path.sep + output_prefix + '.input-label.label.' + str(batch_idx))

    with open(output_folder + os.path.sep + "data.meta.txt", 'w+') as meta_file:
        meta_file.write("signal_length " + str(FLAGS.length) + "\n")
        meta_file.write("file_batch_size " + str(FLAGS.batch) + "\n")
        meta_file.write("normalization " + FLAGS.normalization + "\n")
        meta_file.write("basecall_group " + FLAGS.basecall_group + "\n")
        meta_file.write("basecall_subgroup" + FLAGS.basecall_subgroup + "\n")
        meta_file.write("chromosome" + FLAGS.chrom + "\n")
        meta_file.write("chromosome_start" + FLAGS.chrom_start + "\n")
        meta_file.write("chromosome_end" + FLAGS.chrom_end + "\n")
        if FLAGS.replace:
            meta_file.write("DNA_base A-0 C-1 G-2 T-3" + "\n")
        else:
            meta_file.write("DNA_base A-0 C-1 G-2 T-3 M-4" + "\n")
        meta_file.write("data_type " + FLAGS.mode + "\n")

    return group_src,group_tgt


def build_and_save_dictionary(
    train_path, output_path, num_workers, freq_threshold, max_words
):
    dict = build_dictionary([train_path], num_workers)
    dict.finalize(threshold=freq_threshold, nwords=max_words)
    dict_path = os.path.join(output_path, "dict.txt")
    dict.save(dict_path)
    return dict_path


def build_dictionary(filenames, workers):
    d = dictionary.Dictionary()
    for filename in filenames:
        Tokenizer.add_file_to_dictionary(filename, d, tokenize_line, workers)
    return d


def binarize(args, filename, dict, output_prefix, lang, offset, end):
    ds = indexed_dataset.IndexedDatasetBuilder(
        dataset_dest_file(args, output_prefix, lang, "bin")
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Tokenizer.binarize(filename, dict, consumer, offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def binarize_with_load(args, filename, dict_path, output_prefix, lang, offset, end):
    dict = dictionary.Dictionary.load(dict_path)
    binarize(args, filename, dict, output_prefix, lang, offset, end)
    return dataset_dest_prefix(args, output_prefix, lang)


def dataset_dest_prefix(args, output_prefix, lang):
    base = f"{args.destdir}/{output_prefix}"
    lang_part = (
        f".{args.source_lang}-{args.target_lang}.{lang}" if lang is not None else ""
    )
    return f"{base}{lang_part}"


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return f"{base}.{extension}"


def get_offsets(input_file, num_workers):
    return Tokenizer.find_offsets(input_file, num_workers)


def merge_files(files, outpath):
    ds = indexed_dataset.IndexedDatasetBuilder("{}.bin".format(outpath))
    for file in files:
        ds.merge_file_(file)
        os.remove(indexed_dataset.data_file_path(file))
        os.remove(indexed_dataset.index_file_path(file))
    ds.finalize("{}.idx".format(outpath))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.batch = int(args.batch)
    args.chrom_start = int(args.chrom_start)
    args.chrom_end = int(args.chrom_end)
    main(args)
