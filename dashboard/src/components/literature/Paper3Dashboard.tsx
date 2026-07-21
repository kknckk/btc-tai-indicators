'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, BarChart, Bar, Cell
} from 'recharts';
import { Activity, TrendingUp, CheckCircle, AlertCircle, Layers } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper3Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper3', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie wyników analitycznych (ARDL & SOPR)...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 3.</div>;

  const { adf_tests = {}, short_term_ardl = [], long_term_multipliers = {}, time_series = [], sopr_column = 'SOPR' } = data;

  // Przygotowanie danych mnożników długoterminowych pod wykres słupkowy
  const ltmData = Object.entries(long_term_multipliers).map(([key, val]: [string, any]) => ({
    name: key === sopr_column ? 'SOPR (Zyskowność)' : key === 'PM_proxy' ? 'Puell Multiple (Górnicy)' : 'Adresy Aktywne',
    rawKey: key,
    multiplier: typeof val === 'number' ? val : 0
  }));

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <Layers className="w-6 h-6 text-blue-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">Metodologia ARDL & Kointegracja Pesaran Bounds</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Artykuł bada długoterminową i krótkoterminową dynamikę pomiędzy wskaźnikiem <span className="text-amber-400 font-semibold">SOPR</span> (Spent Output Profitability Ratio), zyskownością górników i liczbą aktywnych adresów a ceną Bitcoina. Model <span className="text-blue-400 font-mono">ARDL (Autoregressive Distributed Lag)</span> pozwala udowodnić istnienie kointegracji w warunkach mieszanej stacjonarności szeregów <span className="font-mono text-xs">I(0)/I(1)</span>.
            </p>
          </div>
        </div>
      </div>

      {/* 1. Tabela testów ADF */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-emerald-400 mb-4 flex items-center gap-2">
          <CheckCircle className="w-5 h-5" />
          Testy Stacjonarności Augmented Dickey-Fuller (ADF)
        </h3>
        <p className="text-slate-400 text-sm mb-4">Sprawdzenie stopnia zintegrowania szeregów przed budową modelu ARDL.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Cena BTC (PriceUSD)</div>
            <div className="flex items-baseline justify-between mt-2">
              <div>
                <span className="text-2xl font-bold font-mono text-white">{adf_tests.PriceUSD_ADF_stat?.toFixed(4) ?? 'N/A'}</span>
                <span className="text-xs text-slate-500 ml-2">(Statystyka ADF)</span>
              </div>
              <span className={`px-2 py-1 rounded text-xs font-semibold ${adf_tests.PriceUSD_p_value < 0.05 ? 'bg-emerald-900/30 text-emerald-400 border border-emerald-800' : 'bg-amber-900/30 text-amber-400 border border-amber-800'}`}>
                p = {adf_tests.PriceUSD_p_value?.toFixed(4) ?? 'N/A'} {adf_tests.PriceUSD_p_value > 0.05 ? '(Niestacjonarny I(1))' : '(Stacjonarny)'}
              </span>
            </div>
          </div>

          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Wskaźnik SOPR ({sopr_column})</div>
            <div className="flex items-baseline justify-between mt-2">
              <div>
                <span className="text-2xl font-bold font-mono text-white">{adf_tests.sopr_ADF_stat?.toFixed(4) ?? 'N/A'}</span>
                <span className="text-xs text-slate-500 ml-2">(Statystyka ADF)</span>
              </div>
              <span className={`px-2 py-1 rounded text-xs font-semibold ${adf_tests.sopr_p_value < 0.05 ? 'bg-emerald-900/30 text-emerald-400 border border-emerald-800' : 'bg-amber-900/30 text-amber-400 border border-amber-800'}`}>
                p = {adf_tests.sopr_p_value?.toFixed(4) ?? 'N/A'} {adf_tests.sopr_p_value < 0.05 ? '(Stacjonarny I(0))' : '(Niestacjonarny I(1))'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Wykres time-series SOPR vs PriceUSD */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-blue-400 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          SOPR a Cena Bitcoina (Sygnały kapitulacji i euforii)
        </h3>
        <p className="text-slate-400 text-sm mb-6">Linia przerywana na poziomie <span className="text-amber-400 font-mono font-bold">1.0</span> wyznacza próg zyskowności (SOPR {'>'} 1 oznacza sprzedaż z zyskiem, SOPR {'<'} 1 kapitulację ze stratą).</p>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={time_series} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('-')[0]} />
              <YAxis yAxisId="left" stroke="#3b82f6" domain={['auto', 'auto']} tickFormatter={(val) => `$${(val/1000).toFixed(0)}k`} />
              <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" domain={[0.9, 1.15]} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} />
              <ReferenceLine yAxisId="right" y={1.0} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'Próg 1.0', fill: '#ef4444', position: 'insideTopRight' }} />
              <Line yAxisId="left" type="monotone" dataKey="PriceUSD" stroke="#3b82f6" dot={false} strokeWidth={2} name="Cena USD" />
              <Line yAxisId="right" type="monotone" dataKey="SOPR" stroke="#f59e0b" dot={false} strokeWidth={1.5} name="SOPR" opacity={0.85} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Mnożniki długoterminowe i krótkoterminowe ARDL */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Tabela krótkoterminowa */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg flex flex-col justify-between">
          <div>
            <h3 className="font-bold text-xl text-amber-400 mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Krótkoterminowa Dynamika ARDL (OLS)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-xs uppercase">
                    <th className="py-2.5 px-3">Zmienna (Opóźnienie)</th>
                    <th className="py-2.5 px-3">Współczynnik</th>
                    <th className="py-2.5 px-3">P-value</th>
                  </tr>
                </thead>
                <tbody>
                  {short_term_ardl.map((item: any, idx: number) => {
                    const isSig = item.p_value < 0.05;
                    return (
                      <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/20 text-sm">
                        <td className="py-2.5 px-3 font-medium text-slate-300">{item.feature}</td>
                        <td className={`py-2.5 px-3 font-mono ${isSig ? (item.coef > 0 ? 'text-emerald-400' : 'text-red-400') : 'text-slate-400'}`}>
                          {item.coef?.toFixed(5) ?? 'N/A'}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-slate-400">
                          {item.p_value?.toFixed(4) ?? 'N/A'}
                          {isSig && <span className="ml-2 px-1 py-0.5 bg-emerald-900/40 text-emerald-400 text-[10px] rounded">Istotne</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Mnożniki Długoterminowe (BarChart) */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
          <h3 className="font-bold text-xl text-purple-400 mb-4 flex items-center gap-2">
            <Layers className="w-5 h-5" />
            Mnożniki Długoterminowe (Long-Term Multipliers)
          </h3>
          <p className="text-slate-400 text-xs mb-4">Wartości wskazują na elastyczność długoterminową ceny wobec zmian wskaźników on-chain.</p>
          <div className="h-[260px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ltmData} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#64748b" />
                <YAxis type="category" dataKey="name" stroke="#cbd5e1" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} />
                <Bar dataKey="multiplier" name="Mnożnik długoterminowy" radius={[0, 4, 4, 0]}>
                  {ltmData.map((entry: any, idx: number) => (
                    <Cell key={`cell-${idx}`} fill={entry.multiplier > 0 ? '#38bdf8' : '#f43f5e'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
