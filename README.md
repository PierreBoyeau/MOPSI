# rare_words_DNA
#Synopsis 
Objectives of the project: find tools to quantify and find exceptional subsequences of nucleotides (abnormaly rare or frequent) in a DNA sequence, potentially very long.
This procedure can be interesting to biologists because exceptional DNA motifs sometimes have biological purpose (see Chi motifs and restriction sites)

#Code
The code offer 2 different ways to find such motifs, both using Markov approximations.
The first one, Counter_simple model the DNA as a 1-Markov Chain.
The second one, Counter_m model the DNA as a Markov Chain of order k-2, where k is the lenght of the motifs studied.

To use one of these class, don't forget to:
- Initialize the lenght of the motifs you want to study
- Specify what DNA sequence you are working on (learn method)

#References
[1] S. Robin, F. Rodolphe, S. Schbath ADN, mots et modèles 2003.

[2] J.F. Delmas, B. Jourdain Modèles aléatoires Mathematiques et Applications 57, 2007.

[3] G. Nuel Significance Score of Motifs in Biological Sequences Bioinformatics: Trends and Methodologies Intech
2011; 978-53. Relations.
