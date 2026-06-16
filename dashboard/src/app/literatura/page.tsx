export default function LiteraturaPage() {
  return (
    <div className="max-w-4xl">
      <h1 className="text-3xl font-bold mb-4">Analizy Oparte o Literaturę Naukową</h1>
      <p className="mb-4 text-gray-700">
        W tej sekcji znajdziesz wdrożenia 10 zaawansowanych artykułów naukowych z zakresu analizy On-Chain Bitcoina.
        Zastosowano tutaj modele ekonometryczne (GARCH, ARDL) oraz modele uczenia maszynowego (Random Forest, SVM, Prophet, PCA).
      </p>
      <p className="mb-4 text-gray-700">
        Modele są przetrenowywane lokalnie w trybie batchowym, a ich wyniki serwowane przez nasze API. 
        Dla każdej pracy wdrożyliśmy proces **weryfikacji tezy**, aby sprawdzić czy wyniki opisywane przez badaczy mają zastosowanie 
        dla najnowszych danych.
      </p>
      <div className="p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
        <p className="font-semibold text-blue-800">Instrukcja:</p>
        <p className="text-blue-700">Wybierz interesującą Cię pracę z menu po lewej stronie, aby przejrzeć tezę, metodologię oraz wygenerowane na żywo wyniki z modeli statystycznych.</p>
      </div>
    </div>
  );
}
