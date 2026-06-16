import React from 'react';
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
} from 'recharts';

export default function Paper1View({ data }: { data: any }) {
  if (!data) return null;

  // Format the correlation matrix into an array for rendering
  const metrics = Object.keys(data.correlation_heatmap);
  
  // Find the 90th percentile of h_t to highlight high volatility periods
  const h_t_values = data.h_t_series.map((d: any) => d.h_t).filter((v: number) => !isNaN(v));
  h_t_values.sort((a: number, b: number) => a - b);
  const percentile90 = h_t_values[Math.floor(h_t_values.length * 0.9)] || 0;

  // Enhance series data with a "high_volatility" background indicator
  const chartData = data.h_t_series.map((d: any) => ({
    ...d,
    time: d.time.split('T')[0], // format date
    high_vol_zone: d.h_t > percentile90 ? Math.max(...data.h_t_series.map((x:any)=>x.PriceUSD||0)) : null,
  }));

  const formatPrice = (val: number) => {
    if (val > 1000) return `$${(val / 1000).toFixed(1)}k`;
    return `$${val}`;
  };

  const getHeatmapColor = (value: number) => {
    // Red for 1, white for 0, blue for -1 (roughly)
    if (value === 1) return 'bg-slate-800 text-white';
    if (value > 0.8) return 'bg-blue-600 text-white';
    if (value > 0.6) return 'bg-blue-400 text-white';
    if (value > 0.4) return 'bg-blue-200 text-black';
    if (value > 0.2) return 'bg-blue-100 text-black';
    if (value < -0.6) return 'bg-red-400 text-white';
    return 'bg-gray-50 text-black';
  };

  return (
    <div className="space-y-8 mt-6">
      <div className="bg-white p-6 border rounded-xl shadow-sm">
        <h3 className="text-xl font-bold mb-4 text-slate-800">1. Analiza Regresji: Wpływ Aktywności na Zmienność</h3>
        <p className="text-gray-600 mb-6 text-sm">
          Poniższa tabela przedstawia współczynniki z regresji OLS, gdzie zmienną zależną jest warunkowa wariancja (h_t) z modelu GARCH(1,1), a zmiennymi objaśniającymi są metryki sieciowe. 
          Wartości p-value poniżej 0.05 oznaczają statystyczną istotność.
        </p>
        
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="py-3 px-4 border-b font-semibold text-left rounded-tl-lg">Wskaźnik Sieciowy</th>
                <th className="py-3 px-4 border-b font-semibold text-right">Współczynnik (Coef)</th>
                <th className="py-3 px-4 border-b font-semibold text-right">P-Value</th>
                <th className="py-3 px-4 border-b font-semibold text-center rounded-tr-lg">Istotność & Wniosek</th>
              </tr>
            </thead>
            <tbody>
              {data.regressions.map((reg: any, i: number) => (
                <tr key={reg.metric} className="hover:bg-slate-50 border-b last:border-0">
                  <td className="py-3 px-4 font-medium text-slate-800">{reg.metric}</td>
                  <td className="py-3 px-4 text-right font-mono text-slate-600">{reg.coef.toFixed(6)}</td>
                  <td className="py-3 px-4 text-right font-mono text-slate-600">{reg.p_value.toFixed(5)}</td>
                  <td className="py-3 px-4 text-center font-bold">
                    {reg.significant ? (
                      reg.coef > 0 ? (
                        <span className="bg-green-100 text-green-700 px-3 py-1.5 rounded-full inline-block min-w-[120px]">Potwierdza (+)</span>
                      ) : (
                        <span className="bg-yellow-100 text-yellow-700 px-3 py-1.5 rounded-full inline-block min-w-[120px]">Odwrotna (-)</span>
                      )
                    ) : (
                      <span className="bg-red-100 text-red-700 px-3 py-1.5 rounded-full inline-block min-w-[120px]">Nieistotny</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white p-6 border rounded-xl shadow-sm">
        <h3 className="text-xl font-bold mb-4 text-slate-800">2. Cena vs Przewidywana Zmienność (GARCH h_t)</h3>
        <p className="text-gray-600 mb-6 text-sm">
          Wykres przedstawia cenę Bitcoina oraz warunkową wariancję (h_t). Szare obszary w tle oznaczają dni ekstremalnej zmienności (h_t w górnym 10. percentylu). Zauważ jak skoki "h_t" często korelują z dużymi ruchami rynkowymi.
        </p>
        
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="time" minTickGap={30} tick={{fill: '#64748b', fontSize: 12}} />
              <YAxis 
                yAxisId="left" 
                tickFormatter={formatPrice} 
                tick={{fill: '#64748b', fontSize: 12}} 
                domain={['auto', 'auto']}
              />
              <YAxis 
                yAxisId="right" 
                orientation="right" 
                tick={{fill: '#64748b', fontSize: 12}} 
                domain={[0, 'auto']}
              />
              <Tooltip 
                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                labelStyle={{ fontWeight: 'bold', color: '#0f172a' }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              
              {/* Highlight background for high volatility */}
              <Area 
                yAxisId="left" 
                type="step" 
                dataKey="high_vol_zone" 
                fill="#f1f5f9" 
                stroke="none" 
                name="Ekstremalna Zmienność" 
                activeDot={false}
              />
              
              <Line 
                yAxisId="left" 
                type="monotone" 
                dataKey="PriceUSD" 
                stroke="#0f172a" 
                strokeWidth={2}
                dot={false}
                name="Cena BTC ($)"
              />
              <Line 
                yAxisId="right" 
                type="monotone" 
                dataKey="h_t" 
                stroke="#ef4444" 
                strokeWidth={2}
                dot={false}
                name="Wariancja (h_t)"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white p-6 border rounded-xl shadow-sm">
        <h3 className="text-xl font-bold mb-4 text-slate-800">3. Macierz Korelacji Miast Aktywności</h3>
        <p className="text-gray-600 mb-6 text-sm">
          Mapa cieplna (Heatmap) pokazująca współzależność między poszczególnymi miarami on-chain. Zgodnie z tezą artykułu, miary aktywności są ze sobą mocno skorelowane.
        </p>
        
        <div className="overflow-x-auto">
          <table className="min-w-max border-collapse">
            <thead>
              <tr>
                <th className="p-3 border text-slate-500 bg-slate-50"></th>
                {metrics.map(m => (
                  <th key={m} className="p-3 border font-semibold text-slate-700 bg-slate-50 text-center">{m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.map(row => (
                <tr key={row}>
                  <th className="p-3 border font-semibold text-slate-700 bg-slate-50 text-left">{row}</th>
                  {metrics.map(col => {
                    const val = data.correlation_heatmap[row][col];
                    return (
                      <td key={col} className={`p-4 border text-center font-mono text-sm ${getHeatmapColor(val)}`}>
                        {val.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
