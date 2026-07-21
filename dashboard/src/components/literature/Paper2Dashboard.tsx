'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis
} from 'recharts';
import { Activity, TrendingUp, Anchor } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper2Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper2', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl">Ładowanie danych analitycznych (NetFlow)...</div>;
  if (error || !data) return <div className="text-red-400 p-8">Błąd podczas ładowania danych dla Paper 2.</div>;

  const { returns_model, volatility_model, time_series = [], scatter_data = [] } = data;

  const regressions = [
    {
      target: 'Zwrot dzienny (r_t)',
      feature: 'NetFlowUSD (Lag 0)',
      coef: returns_model?.coef_lag0 ?? 0,
      p_value: returns_model?.pval_lag0 ?? 1,
    },
    {
      target: 'Zwrot dzienny (r_t)',
      feature: 'NetFlowUSD (Lag 1)',
      coef: returns_model?.coef_lag1 ?? 0,
      p_value: returns_model?.pval_lag1 ?? 1,
    },
    {
      target: 'Zmienność (rv_t)',
      feature: 'NetFlowUSD (Lag 0)',
      coef: volatility_model?.coef_lag0 ?? 0,
      p_value: volatility_model?.pval_lag0 ?? 1,
    },
    {
      target: 'Zmienność (rv_t)',
      feature: 'NetFlowUSD (Lag 1)',
      coef: volatility_model?.coef_lag1 ?? 0,
      p_value: volatility_model?.pval_lag1 ?? 1,
    },
  ];

  return (
    <div className="space-y-8">
      {/* 1. Wykres time-series NetFlowUSD vs PriceUSD */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-blue-400 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Net Flows (USD) a Cena BTC
        </h3>
        <p className="text-slate-400 text-sm mb-6">Napływy netto na giełdy nałożone na cenę Bitcoina.</p>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={time_series} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('-')[0]} />
              <YAxis yAxisId="left" stroke="#3b82f6" domain={['auto', 'auto']} tickFormatter={(val) => `$${(val/1000).toFixed(0)}k`} />
              <YAxis yAxisId="right" orientation="right" stroke="#ef4444" tickFormatter={(val) => `${(val/1e6).toFixed(0)}M`} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }}
              />
              <Line yAxisId="left" type="monotone" dataKey="PriceUSD" stroke="#3b82f6" dot={false} strokeWidth={2} name="Cena USD" />
              <Line yAxisId="right" type="monotone" dataKey="NetFlowUSD" stroke="#ef4444" dot={false} strokeWidth={1} name="Net Flow (USD)" opacity={0.7} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. Tabela Wpływu */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-emerald-400 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Wpływ Net Flows na Zwroty i Zmienność (OLS)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-sm uppercase">
                <th className="py-3 px-4">Zmienna Zależna (Cel)</th>
                <th className="py-3 px-4">Zmienna Niezależna</th>
                <th className="py-3 px-4">Współczynnik</th>
                <th className="py-3 px-4">P-value</th>
              </tr>
            </thead>
            <tbody>
              {regressions.map((reg: unknown, idx: number) => {
                const isSig = reg.p_value < 0.05;
                return (
                  <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/20">
                    <td className="py-3 px-4 font-medium text-slate-300">{reg.target}</td>
                    <td className="py-3 px-4 text-slate-400">{reg.feature}</td>
                    <td className={`py-3 px-4 font-mono ${isSig ? 'text-emerald-400' : 'text-slate-400'}`}>
                      {reg.coef.toFixed(6)}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-400">
                      {reg.p_value.toFixed(4)}
                      {isSig && <span className="ml-2 px-1 bg-emerald-900/50 text-emerald-400 text-xs rounded">Istotne</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Wykres Scatter */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-amber-400 mb-4 flex items-center gap-2">
          <Anchor className="w-5 h-5" />
          NetFlowUSD vs Dzisiejszy Zwrot
        </h3>
        <p className="text-slate-400 text-sm mb-6">Rozrzut dzisiejszego wpływu netto vs dzisiejszy zwrot z inwestycji (r_t).</p>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" dataKey="NetFlowUSD" name="NetFlow USD" stroke="#64748b" tickFormatter={(val) => `${(val/1e6).toFixed(0)}M`} />
              <YAxis type="number" dataKey="r_t" name="Zwrot" stroke="#64748b" tickFormatter={(val) => `${(val * 100).toFixed(1)}%`} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }} />
              <Scatter name="Dni" data={(scatter_data.length > 0 ? scatter_data : time_series).filter((_:any, i:number) => i % 3 === 0)} fill="#f59e0b" fillOpacity={0.6} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
