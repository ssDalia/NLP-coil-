{ 
  mkShell,
  python3,
}: 
mkShell {
  buildInputs = [
    (python3.withPackages (pypkgs: with pypkgs; [
      pandas
      spacy
      spacy-models.en_core_web_sm
    ]))
  ];
}

