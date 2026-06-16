'use client';

import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import Paper1View from '@/components/literature/Paper1View';

// Słownik metadanych i rozszerzonych opisów dla poszczególnych prac naukowych
const paperDetails = {
  '1': {
    title: 'Bitcoin Network Activity and Bitcoin Price Return Volatility (2023)',
    thesis: 'Aktywność sieciowa użytkowników Bitcoina (metryki On-Chain) jest silnie współzależna i pozytywnie związana z dzienną zmiennością zwrotów (Volatility). Możemy traktować ruch w sieci jako bezpośrednie proxy (wskaźnik zastępczy) dla intensywności handlu i docierania nowych informacji na rynek.',
    explanation: 'Co budujemy? Ekonometryczny model warunkowej wariancji GARCH(1,1), do którego "wstrzykujemy" wskaźniki takie jak liczba aktywnych adresów (AdrActCnt) czy zapłacone opłaty (FeeTotUSD). \nDlaczego? Klasyczna analiza techniczna patrzy tylko na cenę. Dodając On-Chain, zaglądamy w "bebechy" Bitcoina. Jeśli nagle rośnie liczba transakcji, model GARCH natychmiast przewidzi zwiększoną zmienność (np. duże skoki cenowe). \nPrzewaga: Wcześniejsze wykrywanie "szoków" rynkowych na podstawie tłoku w sieci, zanim cena drastycznie tąpnie w górę lub w dół.',
    methodology: 'Model GARCH(1,1) dopasowuje warunkową wariancję, którą następnie regresujemy względem logarytmowanych wskaźników aktywności (OLS). Sprawdzamy czy p-value < 0.05 i czy znak kierunkowy (współczynnik) jest dodatni.',
    apiEndpoint: '/api/v1/literature/econometrics'
  },
  '2': {
    title: 'Return and Volatility Forecasting Using On-Chain Flows',
    thesis: 'Przepływy netto (Net Flows) na giełdy posiadają moc predykcyjną dotyczącą bardzo krótkoterminowych stóp zwrotu i nadchodzącej zmienności cenowej.',
    explanation: 'Co budujemy? Liniowe modele regresji zwrotów i zmienności oparte na opóźnionych wartościach NetFlow (Wpłaty minus Wypłaty z giełd). \nDlaczego? Inwestorzy przeważnie przesyłają BTC na giełdę (inflow), gdy chcą go sprzedać, co tworzy presję podażową. Z kolei wypłacanie (outflow) oznacza akumulację w chłodniach (cold wallets). \nPrzewaga: Przewidujemy ruch ceny w ciągu kolejnych 24-48 godzin dzięki obserwowaniu rzeczywistych, twardych ruchów kapitału instytucjonalnego i detalicznego zanim nastąpi kliknięcie "Sell" w arkuszu zleceń.',
    methodology: 'Regresja OLS dla zmiennej zależnej log-zwrotów (oraz wariancji) bazująca na 1- i 2-dniowych opóźnieniach wyliczonej miary NetFlowUSD.',
    apiEndpoint: '/api/v1/literature/econometrics'
  },
  '4': {
    title: 'Using On-Chain Data to Predict Cryptocurrency Cycles (2025)',
    thesis: 'Makroekonomiczne wskaźniki zysku i straty całej sieci (takie jak NUPL czy MVRV Z-Score) w sposób deterministyczny wyznaczają rynkowe cykle euforii i kapitulacji.',
    explanation: 'Co budujemy? Algorytmiczną strategię inwestycyjną typu kontrariańskiego. Algorytm kupuje BTC tylko w strefach "strachu" (MVRV na dnie) i wychodzi z rynku w strefach "chciwości". \nDlaczego? Trzymanie (Buy and Hold) naraża inwestora na potężne spadki (Drawdowns rzędu 80%). Wskaźniki on-chain informują nas o tym, kiedy rynek jest "przegrzany" – niezrealizowane zyski użytkowników są tak ogromne, że masowa realizacja zysków jest tylko kwestią czasu. \nPrzewaga: Systematyczna redukcja ryzyka utraty kapitału w bessie przy jednoczesnym przechwytywaniu większości zysków podczas hossy (wyższy wskaźnik Sharpe\'a).',
    methodology: 'Wyznaczamy historyczne progi percentylowe (np. dolne 20% i górne 80%) dla MVRV_Z oraz NUPL. Budujemy logiczny automat generujący dzienną krzywą kapitału w relacji do benchmarku B&H.',
    apiEndpoint: '/api/v1/literature/strategies'
  },
  '5': {
    title: 'Price Forecasting Using On-Chain Metrics (Prophet & DL)',
    thesis: 'Wskaźniki wyciągnięte z blockchaina, zwłaszcza "Miner Revenue" (Przychody Górników), Opłaty i Aktywne Adresy znacznie redukują błąd w prognozowaniu ceny na 1 do 7 dni w przód.',
    explanation: 'Co budujemy? Modele addytywne i numeryczne takie jak Prophet czy sieci neuronowe traktujące wskaźniki on-chain jako dodatkowe regresory. \nDlaczego? Modelowanie samej ceny zakłada, że rynki są idealnie wydajne. Niestety (lub stety) rynki krypto podlegają ogromnym manipulacjom. Sieci neuronowe "uczą się" nieoczywistych relacji wielowymiarowych między kosztem wydobycia (HashRate), zyskami górników a presją na rzucanie nowych monet na rynek. \nPrzewaga: Przejście od zgadywania do statystycznego rachunku błędu. Wykorzystanie ekonomiki kopania BTC (Miner Profitability) by dostarczyć precyzyjniejszą prognozę kierunku ceny.',
    methodology: 'Zaawansowany model FB Prophet z sezonowością, dla którego wstrzykujemy zewnętrzne parametry On-Chain (Regressors). Kalkulujemy metryki błędu absolutnego (MAE) oraz średniokwadratowego (RMSE).',
    apiEndpoint: '/api/v1/literature/ml'
  },
  '7': {
    title: 'Machine & Deep Learning Models for Predicting Direction',
    thesis: 'Zestawienie klasycznej Analizy Technicznej (TA) razem z danymi On-Chain (Przepływy, Wycena, Górnicy) w algorytmach maszynowych takich jak SVM czy Random Forest osiąga najwyższą skuteczność decyzyjną.',
    explanation: 'Co budujemy? Algorytmy klasyfikacyjne AI decydujące czy kolejnego dnia cena pójdzie w Górę (Up) czy w Dół (Down). Oceniamy wagi poszczególnych cech (Feature Importance). \nDlaczego? Żaden wskaźnik używany samotnie nie gwarantuje zysku. Lasy Losowe (Random Forest) łączą decyzje tysięcy mikro-drzew decyzyjnych uwzględniając równolegle sygnały z przepływów giełdowych, zysków i aktywności. \nPrzewaga: Odporność na rynkowy "szum". Eliminacja bezwartościowych sygnałów i pozostawienie tylko tych zmiennych On-Chain, które w 80% przypadków poprawnie klasyfikowały kierunek rynku.',
    methodology: 'Algorytmy SVC (Support Vector Machines) oraz Random Forest Classifier z walidacją krzyżową lub train-test splitem. Wyrzucamy posortowane wagi drzewa decyzyjnego by wskazać "króla" wskaźników.',
    apiEndpoint: '/api/v1/literature/ml'
  }
};

export default function PaperPage() {
  const { paperId } = useParams();
  const idStr = typeof paperId === 'string' ? paperId : (paperId?.[0] || '');
  const details = paperDetails[idStr as keyof typeof paperDetails];

  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (details) {
      fetch(details.apiEndpoint)
        .then(res => res.json())
        .then(d => {
          setData(d);
          setLoading(false);
        })
        .catch(err => {
          console.error(err);
          setLoading(false);
        });
    }
  }, [details]);

  if (!details) {
    return <div className="p-4 text-gray-700">Artykuł o tym identyfikatorze nie posiada jeszcze przygotowanej merytorycznej interpretacji lub nie wchodzi w skład podstron 1, 2, 4, 5, 7.</div>;
  }

  return (
    <div className="max-w-5xl space-y-6">
      <h1 className="text-3xl font-extrabold text-slate-800 border-b pb-2">[{idStr}] {details.title}</h1>
      
      {/* Sekcja Tezy */}
      <div className="bg-slate-50 border-l-4 border-blue-500 p-5 shadow-sm">
        <h3 className="font-bold text-xl text-blue-900 mb-2">Teza Naukowa (Hypothesis)</h3>
        <p className="text-gray-800 leading-relaxed">{details.thesis}</p>
      </div>

      {/* Sekcja Wyjaśnienia Eksperckiego */}
      <div className="bg-blue-50 border-l-4 border-indigo-600 p-5 shadow-sm">
        <h3 className="font-bold text-xl text-indigo-900 mb-2">Rozumienie Implementacji: Co budujemy i dlaczego?</h3>
        {details.explanation.split('\n').map((line, idx) => {
          const isBold = line.includes(':');
          if (isBold) {
            const [boldPart, ...rest] = line.split(':');
            return (
              <p key={idx} className="mb-2 text-gray-800 leading-relaxed">
                <strong className="text-indigo-800">{boldPart}:</strong>{rest.join(':')}
              </p>
            );
          }
          return <p key={idx} className="mb-2 text-gray-800 leading-relaxed">{line}</p>;
        })}
      </div>

      {/* Sekcja Metodologii */}
      <div className="bg-gray-50 border border-gray-200 p-5 rounded-md shadow-sm">
        <h3 className="font-bold text-lg text-slate-700 mb-2">Metodologia Obliczeniowa (Pod maską)</h3>
        <p className="text-gray-600 text-sm leading-relaxed">{details.methodology}</p>
      </div>

      <h2 className="text-2xl font-bold mt-8 mb-4">Weryfikacja na aktualnych danych</h2>
      
      {loading ? (
        <div className="flex items-center space-x-3 text-gray-500">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
          <p>Przeliczanie lub ładowanie wyników z modeli ML...</p>
        </div>
      ) : (
        <div className="bg-white border rounded-lg p-5 shadow-sm">
          {!data && <p className="text-red-500 font-medium">Brak danych dla tego endpointu. Uruchom skrypty w analytical_pipeline na serwerze.</p>}
          
          {data && idStr === '1' && data.paper1 && (
            <Paper1View data={data.paper1} />
          )}

          {data && idStr === '4' && data.paper4 && (
            <div>
              <h4 className="font-bold mb-3 text-slate-800">Rywalizacja Strategii: Czyste Trzymanie vs Algorytmy On-Chain</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="p-4 border rounded bg-slate-50">
                  <h5 className="font-semibold text-slate-700 border-b pb-2 mb-2">Pasywne (Buy & Hold)</h5>
                  <p className="text-gray-600">Sharpe Ratio (Zysk/Ryzyko): <span className="font-bold text-slate-800">{data.paper4.metrics.buy_and_hold.sharpe.toFixed(2)}</span></p>
                  <p className="text-gray-600">Max Drawdown (Upadek kapitału): <span className="font-bold text-red-600">{(data.paper4.metrics.buy_and_hold.max_drawdown*100).toFixed(1)}%</span></p>
                </div>
                <div className="p-4 border rounded bg-blue-50 border-blue-200">
                  <h5 className="font-semibold text-blue-900 border-b border-blue-200 pb-2 mb-2">Model (MVRV + NUPL)</h5>
                  <p className="text-blue-800">Sharpe Ratio (Zysk/Ryzyko): <span className="font-bold text-blue-900">{data.paper4.metrics.strategy.sharpe.toFixed(2)}</span></p>
                  <p className="text-blue-800">Max Drawdown (Upadek kapitału): <span className="font-bold text-green-700">{(data.paper4.metrics.strategy.max_drawdown*100).toFixed(1)}%</span></p>
                </div>
              </div>
              <p className="text-sm text-gray-500">Zgodnie z tezą pracy, lepsza strategia kontrariańska powinna generować istotnie niższy upadek kapitału podczas tzw. krypto zimy (Bear Market) oraz zapewniać wyższą jakość zwrotów z ryzyka (Wyższy Sharpe).</p>
            </div>
          )}

          {data && idStr === '5' && data.paper5 && (
            <div>
              <h4 className="font-bold mb-3 text-slate-800">Weryfikacja modelu predykcyjnego FB Prophet z Regresorami On-Chain</h4>
              <div className="bg-indigo-50 border border-indigo-100 p-4 rounded inline-block mb-4">
                <p className="text-indigo-900 font-semibold">Błąd Średniokwadratowy (RMSE): {data.paper5.metrics.RMSE.toFixed(2)}</p>
                <p className="text-indigo-900 font-semibold">Błąd Absolutny (MAE): {data.paper5.metrics.MAE.toFixed(2)}</p>
              </div>
              <p className="text-sm text-gray-600">Model oparty na "Miner Revenue" oraz "Zliczaniu Adresów" uczy się sezonowości Bitcoina minimalizując margines wpadki pomiędzy faktycznym a prognozowanym stanem rzeczy w czasie.</p>
            </div>
          )}

          {data && idStr === '7' && data.paper7 && (
            <div>
              <h4 className="font-bold mb-3 text-slate-800">Sztuczna Inteligencja decydująca o kierunku Bitcoina</h4>
              <div className="flex space-x-6 mb-6">
                <div className="bg-slate-50 p-4 border rounded shadow-sm w-1/2">
                  <p className="text-gray-500 text-sm uppercase tracking-wider mb-1">Skuteczność (Accuracy)</p>
                  <p className="text-2xl font-bold text-indigo-700">{(data.paper7.random_forest.accuracy*100).toFixed(1)}%</p>
                </div>
                <div className="bg-slate-50 p-4 border rounded shadow-sm w-1/2">
                  <p className="text-gray-500 text-sm uppercase tracking-wider mb-1">Jakość Sygnałów (F1 Score)</p>
                  <p className="text-2xl font-bold text-indigo-700">{(data.paper7.random_forest.f1*100).toFixed(1)}%</p>
                </div>
              </div>
              
              <h4 className="font-semibold mt-4 mb-2 text-slate-700 border-b pb-2">Jakich wskaźników Algorytm RF używa najchętniej? (Top Feature Importance)</h4>
              <ul className="list-decimal ml-6 space-y-2">
                {data.paper7.feature_importance.slice(0, 5).map((f: any, i: number) => (
                  <li key={f.feature} className="text-gray-700">
                    Wskaźnik <span className="font-bold text-slate-900">{f.feature}</span> ma wagę decyzyjną <span className="font-semibold text-green-700">{(f.importance * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
              <p className="text-sm text-gray-500 mt-6">Zgodnie z tezą artykułów ML, dodanie danych z sieci on-chain drastycznie spycha popularne wskaźniki cenowe "TA" na dalszy plan, czyniąc on-chain mózgiem operacyjnym sieci neuronowych.</p>
            </div>
          )}

        </div>
      )}
    </div>
  );
}
