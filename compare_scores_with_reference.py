import argparse
import json
import os
import shutil

def generate_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=str, required=True)
    parser.add_argument('--scores', type=str, required=True)
    return parser

def main(args):
    with open(args.reference, 'r') as f:
        reference = json.load(f)
    with open(args.scores, 'r') as f:
        scores = json.load(f)
    
    ref_total_streamlines = reference['total_streamlines']
    ref_VS = reference['VS']
    ref_VB = reference['VB']
    ref_VS_ratio = reference['VS_ratio']
    ref_IS = reference['IS']
    ref_IS_ratio = reference['IS_ratio']
    ref_mean_f1 = reference['mean_f1']

    sco_total_streamlines = scores['total_streamlines']
    sco_VS = scores['VS']
    sco_VB = scores['VB']
    sco_VS_ratio = scores['VS_ratio']
    sco_IS = scores['IS']
    sco_IS_ratio = scores['IS_ratio']
    sco_mean_f1 = scores['mean_f1']


    comp_string = ""

    comp_string += "Reference vs Scores ({} and {})\n".format(args.reference, args.scores)
    comp_string += 'streamlines: {:.3f} vs {:.3f},\tdelta: {:.3f}\n'.format(ref_total_streamlines, sco_total_streamlines, sco_total_streamlines - ref_total_streamlines)
    comp_string += 'VS:          {:.3f} vs {:.3f},\tdelta: {:.3f}\n'.format(ref_VS, sco_VS, sco_VS - ref_VS)
    comp_string += 'VB:          {:.3f} vs {:.3f},\t\tdelta: {:.3f}\n'.format(ref_VB, sco_VB, sco_VB - ref_VB)
    comp_string += 'VS_ratio:    {:.3f} vs {:.3f},\t\tdelta: {:.3f},\timprovement: {:.3f}%\n'.format(ref_VS_ratio, sco_VS_ratio, sco_VS_ratio - ref_VS_ratio, (sco_VS_ratio - ref_VS_ratio) * 100)
    comp_string += 'IS:          {:.3f} vs {:.3f},\tdelta: {:.3f}\n'.format(ref_IS, sco_IS, sco_IS - ref_IS)
    comp_string += 'IS_ratio:    {:.3f} vs {:.3f},\t\tdelta: {:.3f},\timprovement: {:.3f}%\n'.format(ref_IS_ratio, sco_IS_ratio, sco_IS_ratio - ref_IS_ratio, (sco_IS_ratio - ref_IS_ratio) * 100)
    comp_string += 'mean_f1:     {:.3f} vs {:.3f},\t\tdelta: {:.3f},\timprovement: {:.3f}%\n'.format(ref_mean_f1, sco_mean_f1, sco_mean_f1 - ref_mean_f1, (sco_mean_f1 - ref_mean_f1) * 100)

    print(comp_string)
    with open(os.path.join(os.path.dirname(args.scores), 'score_comparison.txt'), 'w') as f:
        f.write(comp_string)
    print('Score comparison saved to {}'.format(os.path.join(os.path.dirname(args.scores), 'score_comparison.txt')))
    shutil.copyfile(args.reference, os.path.join(os.path.dirname(args.scores), 'comparison_reference.json'))


if '__main__' == __name__:
    parser = generate_argument_parser()
    main(parser.parse_args())

