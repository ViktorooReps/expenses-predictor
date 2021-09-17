from argparse import ArgumentParser
from pathlib import Path

from data.datamodel import Dataset, Serializable
from models.extractors import name2trainable_extractor, name2extractor
from models.normalizers import name2trainable_normalizer, name2normalizer
from models.predictors import name2trainable_predictor
from models.registered import PredictorName, ExtractorName, NormalizerName

if __name__ == '__main__':
    predictor_names = [pn.value for pn in PredictorName]
    extractor_names = [en.value for en in ExtractorName]
    normalizer_names = [nn.value for nn in NormalizerName]

    parser = ArgumentParser()
    parser.add_argument('dataset_path', type=Path)
    parser.add_argument('predictor_save_path', type=Path)
    parser.add_argument('extractor_save_path', type=Path, default=None)
    parser.add_argument('normalizer_save_path', type=Path, default=None)
    parser.add_argument('-predictor', choices=predictor_names, type=str, default=PredictorName.BASELINE.value)
    parser.add_argument('-extractor', choices=extractor_names, type=str, default=PredictorName.STUB.value)
    parser.add_argument('-normalizer', choices=normalizer_names, type=str, default=NormalizerName.STUB.value)

    args = parser.parse_args()

    predictor_name = PredictorName(args.predictor)
    predictor = name2trainable_predictor[predictor_name](args.predictor_save_path)

    print('Downloading dataset...')
    dataset = Dataset.load(args.dataset_path)

    extractor_name = ExtractorName(args.extractor)
    print(f'Extracting features with {extractor_name} extractor')
    if extractor_name in name2trainable_extractor:
        extractor = name2trainable_extractor[extractor_name](args.extractor_save_path)
        extractor.fit(dataset)
    else:
        extractor = name2extractor[extractor_name]()

    normalizer_name = NormalizerName(args.normalizer)
    print(f'Normalizing features with {normalizer_name} extractor')
    if normalizer_name in name2trainable_normalizer:
        normalizer = name2trainable_normalizer[normalizer_name](args.normalizer_save_path)
        normalizer.fit(dataset)
    else:
        normalizer = name2normalizer[normalizer_name]()

    dataset.modify_users(extractor.extract_users)
    dataset.modify_users(normalizer.normalize_users)

    predictor.fit(dataset)
    predictor.save()
    if isinstance(normalizer, Serializable):
        normalizer.save()
    if isinstance(extractor, Serializable):
        extractor.save()
