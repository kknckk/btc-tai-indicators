'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, ScatterChart, Scatter
} from 'recharts';
import { Flame, CheckCircle, ShieldAlert, Layers, Activity } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper8Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper8', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie analiz kapitulacji i kohort (CDD)...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 8.</div>;

  const { p95_threshold = 0, cdd_column = 'CDD', regression = [], time_series = [], scatter_data = [] } = data;

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <Flame className="w-6 h-6 text-orange-500 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">Ekstrema Coin Days Destroyed (CDD) a Punkty Przegięcia Cykli</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Wskaźnik <span className="text-orange-400 font-semibold">Coin Days Destroyed (CDD)</span> mierzy aktywność długoterminowych posiadaczy (LTH - Long-Term Holders), mnożąc wolumen przesuniętych monet przez liczbę dni ich wcześniejszego uśpienia. Skrajne ekstrema (<span className="font-mono text-xs text-red-400">CDD &gt; 95. percentyla</span>) sygnalizują dystrybucję starych portfeli podczas szczytów hossy lub kapitulację w dołkach bessy.
            </p>
          </div>
        </div>
      </div>

      {/* 1. Kafelek z progiem 95. percentyla i tabela regresji */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-orange-400 uppercase tracking-wider">Próg Ekstremum (95. Percentyl CDD)</span>
            <div className="text-3xl font-bold font-mono text-white mt-2">
              {(p95_threshold / 1e6).toFixed(1)}M <span className="text-xs font-normal text-slate-400">moneto-dni</span>
            </div>
            <p className="text-xs text-slate-400 mt-3 leading-relaxed">
              Przekroczenie tej wartości klasyfikuje dzień jako skrajną dystrybucję długoterminowego kapitału sieci.
            </p>
          </div>
          <div className="mt-4 pt-4 border-t border-slate-800 flex items-center gap-2 text-xs text-orange-400/80">
            <ShieldAlert className="w-4 h-4 flex-shrink-0" />
            <span>Sygnał ostrzegawczy dla rynku</span>
          </div>
        </div>

        {/* Regresja OLS */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg lg:col-span-2 flex flex-col justify-between">
          <div>
            <h3 className="font-bold text-lg text-white mb-3 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
              Wpływ Ekstremów CDD na Zmienność Rynku (Regresja OLS)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-xs uppercase">
                    <th className="py-2.5 px-3">Zmienna Zależna</th>
                    <th className="py-2.5 px-3">Zmienna Wyjaśniająca</th>
                    <th className="py-2.5 px-3">Współczynnik</th>
                    <th className="py-2.5 px-3">P-value</th>
                  </tr>
                </thead>
                <tbody>
                  {regression.map((reg: any, idx: number) => {
                    const isSig = reg.p_value < 0.05;
                    return (
                      <tr key={idx} className="border-b border-slate-800/50 text-sm">
                        <td className="py-2.5 px-3 font-medium text-slate-300">{reg.target}</td>
                        <td className="py-2.5 px-3 text-slate-400 font-mono">{reg.feature}</td>
                        <td className={`py-2.5 px-3 font-mono font-bold ${reg.coef > 0 ? 'text-orange-400' : 'text-blue-400'}`}>
                          {reg.coef?.toFixed(5) ?? 'N/A'}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-slate-400">
                          {reg.p_value?.toFixed(4) ?? 'N/A'}
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
      </div>

      {/* 2. Wykres CDD vs PriceUSD z zaznaczoną linią 95. percentyla */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-orange-400 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Szereg Czasowy CDD a Cena Bitcoina
        </h3>
        <p className="text-slate-400 text-sm mb-6">Linia przerywana wyznacza próg 95. percentyla (<span className="text-red-400 font-mono">{(p95_threshold/1e6).toFixed(1)}M</span>).</p>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={time_series} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('-')[0]} />
              <YAxis yAxisId="left" stroke="#3b82f6" domain={['auto', 'auto']} tickFormatter={(val) => `$${(val/1000).toFixed(0)}k`} />
              <YAxis yAxisId="right" orientation="right" stroke="#f97316" tickFormatter={(val) => `${(val/1e6).toFixed(0)}M`} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} />
              <ReferenceLine yAxisId="right" y={p95_threshold} stroke="#ef4444" strokeDasharray="4 4" label={{ value: '95th Percentile', fill: '#ef4444', position: 'insideTopRight' }} />
              <Line yAxisId="left" type="monotone" dataKey="PriceUSD" stroke="#3b82f6" dot={false} strokeWidth={2} name="Cena USD" />
              <Line yAxisId="right" type="monotone" dataKey="CDD" stroke="#f97316" dot={false} strokeWidth={1} name="CDD" opacity={0.8} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Wykres rozrzutu: CDD vs r_t */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-amber-400 mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5" />
          Rozrzut CDD vs Zwrot Dzienny (r_t)
        </h3>
        <p className="text-slate-400 text-sm mb-6">Wizualizacja reakcji kursu na gwałtowne przebudzenia starych monet.</p>
        <div className="h-[320px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" dataKey="CDD" name="CDD" stroke="#64748b" tickFormatter={(v) => `${(v/1e6).toFixed(0)}M`} />
              <YAxis type="number" dataKey="r_t" name="Zwrot r_t" stroke="#64748b" tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }} />
              <Scatter name="Dni" data={scatter_data.filter((_:any, i:number) => i % 3 === 0)} fill="#f97316" fillOpacity={0.6} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
