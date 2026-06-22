export default function LiteraturaPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-extrabold text-white">Analizy Oparte o Literaturę Naukową</h1>
      <p className="text-slate-400 text-lg leading-relaxed">
        W tej sekcji znajdziesz wdrożenia 10 zaawansowanych artykułów naukowych z zakresu analizy On-Chain Bitcoina.
        Zastosowano tutaj modele ekonometryczne (GARCH, ARDL) oraz modele uczenia maszynowego (Random Forest, SVM, Prophet, PCA).
      </p>
      <p className="text-slate-400 text-lg leading-relaxed">
        Celem tego modułu jest próba odtworzenia oryginalnych tez autorskich. Możesz sprawdzić, co opisuje dany artykuł, jakie metody zostały w nim zaproponowane oraz jakie są wskazówki dotyczące odtworzenia kodu.
      </p>
      <div className="p-6 bg-blue-900/20 border border-blue-800 rounded-xl shadow-lg mt-8">
        <p className="font-semibold text-blue-400 mb-2 text-lg">Instrukcja:</p>
        <p className="text-blue-200">
          Wybierz interesującą Cię pracę z menu po lewej stronie, aby przeczytać podsumowanie, tezę, zalecany stos technologiczny (Tech Stack) do reprodukcji eksperymentów oraz status dostępności oryginalnego kodu autorów.
        </p>
      </div>
    </div>
  );
}
