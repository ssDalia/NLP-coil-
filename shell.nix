{ 
  mkShell,
  python3,
}: 
mkShell {
  buildInputs = [
    (python3.withPackages (pypkgs: with pypkgs; [
      spacy
      spacy-models.en_core_web_sm
      fasttext
      pandas
      numpy
      matplotlib
      seaborn
      scikit-learn
      tqdm
    ]))
  ];
}

