'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceArea
} from 'recharts';
import { Activity, TrendingUp, AlertTriangle } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper1Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper1', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl">Ładowanie danych analitycznych (GARCH)...</div>;
  if (error || !data) return <div className="text-red-400 p-8">Błąd podczas ładowania danych dla Paper 1.</div>;

  const { correlation_heatmap, h_t_series, regressions } = data;

  return (
    <div className="space-y-8">
      {/* 1. Tabela Regresji */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-emerald-400 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Wpływ aktywności na zmienność (OLS na h_t)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-sm uppercase">
                <th className="py-3 px-4">Metryka</th>
                <th className="py-3 px-4">Współczynnik (Coef)</th>
                <th className="py-3 px-4">P-value</th>
                <th className="py-3 px-4">Istotność</th>
              </tr>
            </thead>
            <tbody>
              {regressions.map((reg: any) => {
                const isSigPositive = reg.significant && reg.coef > 0;
                const isSigNegative = reg.significant && reg.coef < 0;
                return (
                  <tr key={reg.metric} className="border-b border-slate-800/50 hover:bg-slate-800/20">
                    <td className="py-3 px-4 font-medium text-slate-300">{reg.metric}</td>
                    <td className={`py-3 px-4 font-mono ${isSigPositive ? 'text-emerald-400' : isSigNegative ? 'text-red-400' : 'text-slate-400'}`}>
                      {reg.coef.toFixed(5)}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-400">{reg.p_value.toFixed(4)}</td>
                    <td className="py-3 px-4">
                      {reg.significant ? (
                        <span className={`px-2 py-1 rounded text-xs font-bold ${isSigPositive ? 'bg-emerald-900/30 text-emerald-400 border border-emerald-800/50' : 'bg-red-900/30 text-red-400 border border-red-800/50'}`}>
                          {isSigPositive ? 'Istotnie Dodatni' : 'Istotnie Ujemny'}
                        </span>
                      ) : (
                        <span className="text-slate-600 text-sm">Brak</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. Wykres PriceUSD vs h_t */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-blue-400 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Cena BTC a przewidywana zmienność (GARCH h_t)
        </h3>
        <p className="text-slate-400 text-sm mb-6">Czerwonym tłem oznaczono okresy ekstremalnej zmienności (h_t {'>'} 90. percentyla).</p>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={h_t_series} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('-')[0]} />
              <YAxis yAxisId="left" stroke="#64748b" domain={['auto', 'auto']} tickFormatter={(val) => `$${(val/1000).toFixed(0)}k`} />
              <YAxis yAxisId="right" orientation="right" stroke="#64748b" hide />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }}
                itemStyle={{ color: '#38bdf8' }}
              />
              {h_t_series.filter((d: any) => d.is_high_vol).map((d: any, idx: number) => (
                <ReferenceArea key={idx} yAxisId="left" x1={d.time} x2={d.time} fill="#ef4444" fillOpacity={0.1} />
              ))}
              <Line yAxisId="left" type="monotone" dataKey="PriceUSD" stroke="#3b82f6" dot={false} strokeWidth={2} name="Cena USD" />
              <Line yAxisId="right" type="monotone" dataKey="h_t" stroke="#ef4444" dot={false} strokeWidth={1} name="h_t (Wariancja)" opacity={0.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* 3. Macierz Korelacji */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-amber-400 mb-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          Heatmapa Korelacji Aktywności Sieciowej
        </h3>
        <p className="text-slate-400 text-sm mb-6">Silne korelacje dodatnie (czerwony) i ujemne (niebieski).</p>
        
        <div className="overflow-x-auto">
          <div className="min-w-max">
            <div className="flex">
              <div className="w-32"></div>
              {Object.keys(correlation_heatmap).map(col => (
                <div key={col} className="w-16 text-center text-xs font-bold text-slate-500 rotate-45 origin-bottom-left -translate-y-4">{col}</div>
              ))}
            </div>
            {Object.entries(correlation_heatmap).map(([row, cols]: [string, any]) => (
              <div key={row} className="flex h-10 items-center">
                <div className="w-32 text-right pr-4 text-xs font-bold text-slate-400 truncate" title={row}>{row}</div>
                {Object.values(cols).map((val: any, idx: number) => {
                  const r = val > 0 ? Math.floor(255 * val) : 0;
                  const b = val < 0 ? Math.floor(255 * Math.abs(val)) : 0;
                  const alpha = Math.abs(val);
                  return (
                    <div 
                      key={idx} 
                      className="w-16 h-10 flex items-center justify-center text-[10px] text-white border border-slate-900"
                      style={{ backgroundColor: `rgba(${r}, 0, ${b}, ${Math.max(0.2, alpha)})` }}
                      title={`${row} x ${Object.keys(cols)[idx]}: ${val.toFixed(2)}`}
                    >
                      {val.toFixed(2)}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
