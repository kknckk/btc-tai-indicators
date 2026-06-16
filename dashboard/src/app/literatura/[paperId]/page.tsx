'use client';

import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';

// Słownik metadanych dla prac
const paperDetails = {
  '1': {
    title: 'Bitcoin Network Activity and Bitcoin Price Return Volatility',
    thesis: 'Aktywność sieciowa użytkowników Bitcoina (on‑chain activity metrics) jest silnie współzależna i dodatnio związana z dzienną zmiennością zwrotów; można ją traktować jako proxy intensywności informacji/handlu.',
    methodology: 'GARCH(1,1) modelujący wariancję zwrotów, a następnie regresja wariancji na wybrane wskaźniki aktywności (np. AdrActCnt, TxCnt). Weryfikujemy czy współczynniki wariancji są dodatnie i istotne (zielone p-value).',
    apiEndpoint: '/api/v1/literature/econometrics'
  },
  '4': {
    title: 'Using On-Chain Data to Predict Bitcoin Cycles',
    thesis: 'Wskaźniki NUPL oraz MVRV Z-score mogą być użyte jako kontrariańskie sygnały wejścia/wyjścia, generując strategie pobijające standardowe Buy-and-Hold.',
    methodology: 'Budujemy strategie opartą na kwantylach wskaźników (np. kupno poniżej 20-percentyla MVRV, sprzedaż powyżej 80-percentyla). Testujemy log-equity curves i Sharpe Ratio.',
    apiEndpoint: '/api/v1/literature/strategies'
  },
  '5': {
    title: 'On-chain analysis and cryptocurrency price forecasting using on-chain metrics',
    thesis: 'Wskaźniki on-chain (szczególnie opłaty, miner revenue i aktywne adresy) poprawiają prognozy krótko-terminowe cen w modelach Prophet i sieciach neuronowych.',
    methodology: 'Modelowanie za pomocą regresorów zewnętrznych w modelu Prophet. Ocena na podstawie zysku MAE/RMSE w stosunku do modelu bazowego.',
    apiEndpoint: '/api/v1/literature/ml'
  },
  '7': {
    title: 'Using Machine and Deep Learning Models... for Predicting Bitcoin Price Direction',
    thesis: 'Modele ML korzystające z kombinacji ceny, TA i on‑chain metrics osiągają wysoką skuteczność w prognozowaniu kierunku ruchów BTC.',
    methodology: 'Klasyfikacja Random Forest oraz SVM przewidująca kierunek zwrotu. Badamy Accuracy, F1 oraz oceniamy wagę cech (Feature Importance).',
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
      // Pobieranie danych z naszego backendu
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
    return <div className="p-4">Artykuł nie został jeszcze zaimplementowany lub ID jest nieprawidłowe. Wdrażanie w toku (zobacz plan w task.md).</div>;
  }

  return (
    <div className="max-w-5xl">
      <h1 className="text-2xl font-bold mb-2">[{idStr}] {details.title}</h1>
      
      <div className="bg-gray-100 p-4 rounded mb-6">
        <h3 className="font-semibold text-lg text-blue-800">Teza (Hypothesis)</h3>
        <p className="text-gray-800">{details.thesis}</p>
      </div>

      <div className="bg-gray-50 border p-4 rounded mb-6">
        <h3 className="font-semibold text-lg text-green-800">Metodologia</h3>
        <p className="text-gray-700">{details.methodology}</p>
      </div>

      <h2 className="text-xl font-bold mb-4 border-b pb-2">Wyniki Modeli (Wygenerowane Lokalnie)</h2>
      
      {loading ? (
        <p>Ładowanie wyników z serwera API...</p>
      ) : (
        <div className="space-y-4">
          {!data && <p className="text-red-500">Brak danych dla tego endpointu. Upewnij się, że skrypty analityczne zakończyły pracę i zapisały JSON w results/.</p>}
          
          {data && idStr === '1' && data.paper1 && (
            <div className="overflow-x-auto">
              <h4 className="font-semibold mb-2">Tabela: Regresja zmienności warunkowej (GARCH) na wskaźniki</h4>
              <table className="min-w-full bg-white border border-gray-300">
                <thead>
                  <tr className="bg-gray-200 text-gray-700">
                    <th className="py-2 px-4 border">Wskaźnik</th>
                    <th className="py-2 px-4 border">Współczynnik</th>
                    <th className="py-2 px-4 border">P-Value</th>
                    <th className="py-2 px-4 border">Istotność statystyczna</th>
                  </tr>
                </thead>
                <tbody>
                  {data.paper1.regressions.map((reg: any) => (
                    <tr key={reg.metric} className="text-center">
                      <td className="py-2 px-4 border font-medium">{reg.metric}</td>
                      <td className="py-2 px-4 border">{reg.coef.toFixed(6)}</td>
                      <td className="py-2 px-4 border">{reg.p_value.toFixed(4)}</td>
                      <td className={`py-2 px-4 border font-bold ${reg.significant && reg.coef > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {reg.significant ? (reg.coef > 0 ? 'Istotny (+)' : 'Istotny (-)') : 'Nieistotny'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-sm text-gray-500 mt-2">Weryfikacja tezy: Jeśli wskaźniki aktywności są 'Istotne (+)' oznacza to pozytywną weryfikację tezy z artykułu.</p>
            </div>
          )}

          {data && idStr === '4' && data.paper4 && (
            <div>
              <h4 className="font-semibold mb-2">Wskaźniki strategii (Od 2018)</h4>
              <ul className="list-disc ml-6 mb-4">
                <li><strong>Buy & Hold Sharpe:</strong> {data.paper4.metrics.buy_and_hold.sharpe.toFixed(2)} | Max DD: {(data.paper4.metrics.buy_and_hold.max_drawdown*100).toFixed(2)}%</li>
                <li><strong>MVRV/NUPL Strategy Sharpe:</strong> {data.paper4.metrics.strategy.sharpe.toFixed(2)} | Max DD: {(data.paper4.metrics.strategy.max_drawdown*100).toFixed(2)}%</li>
              </ul>
              <p className="text-sm text-gray-500">Zgodnie z tezą pracy, skuteczna strategia kontrariańska powinna oferować znacznie wyższy współczynnik Sharpe'a lub niższy Drawdown niż strategia naiwna.</p>
            </div>
          )}

          {data && idStr === '5' && data.paper5 && (
            <div>
              <h4 className="font-semibold mb-2">Predykcja Prophet (RMSE: {data.paper5.metrics.RMSE.toFixed(2)})</h4>
              <p>Model z regresorami zewnętrznymi On-Chain (RevUSD, AdrActCnt).</p>
            </div>
          )}

          {data && idStr === '7' && data.paper7 && (
            <div>
              <h4 className="font-semibold mb-2">Klasyfikacja kierunku (Random Forest)</h4>
              <p>Accuracy: {(data.paper7.random_forest.accuracy*100).toFixed(1)}%</p>
              <h4 className="font-semibold mt-4 mb-2">Top Wagi Cech (Feature Importance):</h4>
              <ul className="list-disc ml-6">
                {data.paper7.feature_importance.slice(0, 5).map((f: any) => (
                  <li key={f.feature} className="text-green-800 font-medium">{f.feature}: {f.importance.toFixed(3)}</li>
                ))}
              </ul>
            </div>
          )}

        </div>
      )}
    </div>
  );
}
