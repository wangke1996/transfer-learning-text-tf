import argparse
import os
import re
import pickle as pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="ACL", help="ACL | Markov | huffman_tree | two_tree")
    parser.add_argument("--data_type", type=str, default="news", help="movie | news | tweet")
    parser.add_argument("--unlabeled_data_nums", nargs='+', type=int, default=[10000, 50000, 150000],
                        help="how many unlabeled data samples was used in pretrain")
    parser.add_argument("--labeled_data_num", type=int, default=8000,
                        help="train data samples for each label")
    parser.add_argument("--method", type=str, default="PVDM", help="PVDM | PVBOW")
    parser.add_argument("--tasks", nargs='+', type=str, default=['01', '02', '03', '04', '05', '012345'])
    args = parser.parse_args()
    args.pre_trained = "doc2vec"

    output_file_path = os.path.join(args.pre_trained, args.data_folder, args.data_type, args.method + '_results')
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    output_file_path = os.path.join(output_file_path, '_'.join(
        [str(x) for x in args.unlabeled_data_nums]) + '_unlabeled_' + str(args.labeled_data_num) + '.csv')

    result_dict = dict()
    for unlabeled_data_num in args.unlabeled_data_nums:
        result_dict[unlabeled_data_num] = dict()
        pre_dir = os.path.join(args.pre_trained, args.data_folder, args.data_type, str(unlabeled_data_num), args.method)
        for task in args.tasks:
            result_dict[unlabeled_data_num][task] = dict()
            file_path = os.path.join(pre_dir, 'bit_'.join(task) + 'bit_' + str(args.labeled_data_num), 'accuracy.txt')
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().splitlines()
                i = -1
                last_line = lines[i].strip()
                while last_line == '':
                    i = i - 1
                    last_line = lines[i].strip()
            precisions = [float(x) for x in
                          re.search(r'[^\[]+$',
                                    re.search(r'precision:[^\]]+', last_line).group()).group().strip().split()]
            recalls = [float(x) for x in
                       re.search(r'[^\[]+$', re.search(r'recall:[^\]]+', last_line).group()).group().strip().split()]
            fscores = [float(x) for x in
                       re.search(r'[^\[]+$', re.search(r'fscore:[^\]]+', last_line).group()).group().strip().split()]
            accuracies = [float(x) for x in
                          re.search(r'[^\[]+$',
                                    re.search(r' accuracy:[^\]]+', last_line).group()).group().strip().split()]
            specificities = [float(x) for x in
                             re.search(r'[^\[]+$',
                                       re.search(r'specificity:[^\]]+', last_line).group()).group().strip().split()]
            TP = [round(args.labeled_data_num * x) for x in recalls]
            FN = [args.labeled_data_num - x for x in TP]
            FP = [round(x / y - x) for (x, y) in zip(TP, precisions)]
            TN = [round(args.labeled_data_num * (len(task) - 1) * x) for x in specificities]
            average_precision = sum(TP) / (sum(TP) + sum(FP))
            average_recall = sum(TP) / (sum(TP) + sum(FN))
            average_fscore = 2 * average_precision * average_recall / (average_precision + average_recall)
            average_specificity = sum(TN) / (sum(TN) + sum(FP))
            average_accuracy = (sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN))

            precisions.append(sum(precisions)/len(precisions))
            recalls.append(sum(recalls) / len(recalls))
            fscores.append(sum(fscores) / len(fscores))
            accuracies.append(sum(accuracies) / len(accuracies))
            specificities.append(sum(specificities) / len(specificities))
            # precisions.append(average_precision)
            # recalls.append(average_recall)
            # fscores.append(average_recall)
            # accuracies.append(average_accuracy)
            # specificities.append(average_specificity)

            all_accuracy = float(re.search(r' .+$', re.search(r'all_accuracy:.+$', last_line).group()).group().strip())
            if len(task) == 2:
                positive_label = 1
            else:
                positive_label = -1
            result_dict[unlabeled_data_num][task]['precision'] = precisions[positive_label]
            result_dict[unlabeled_data_num][task]['recall'] = recalls[positive_label]
            result_dict[unlabeled_data_num][task]['fscore'] = fscores[positive_label]
            result_dict[unlabeled_data_num][task]['specificity'] = specificities[positive_label]
            result_dict[unlabeled_data_num][task]['accuracy'] = accuracies[positive_label]
            result_dict[unlabeled_data_num][task]['all_accuracy'] = all_accuracy
    with open(output_file_path, 'w', encoding='utf8') as f:
        f.write(',')
        for unlabeled_data_num in args.unlabeled_data_nums:
            f.write(',' + str(unlabeled_data_num))
        f.write('\n')
        for task in args.tasks:
            for content in ['precision', 'recall', 'fscore', 'specificity', 'accuracy', 'all_accuracy']:
                f.write(task + 'bit,' + content)
                for unlabeled_data_num in args.unlabeled_data_nums:
                    f.write(',' + str(result_dict[unlabeled_data_num][task][content]))
                f.write('\n')
