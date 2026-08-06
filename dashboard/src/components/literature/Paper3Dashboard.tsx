'use client';

import React from 'react';
import useSWR from 'swr';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell
} from 'recharts';
import { TrendingUp, CheckCircle, Layers, Activity } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper3Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper3', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie wyników analitycznych (ARDL V2)...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 3.</div>;

  const bounds_test = data.bounds_test || {};
  const long_run_multipliers = data.long_run_multipliers || {};
  const ecm_results = data.ecm_results || {};
  const adf_tests = data.adf_kpss || {};

  // Przygotowanie danych mnożników długoterminowych pod wykres słupkowy
  const ltmData = Object.entries(long_run_multipliers).map(([key, val]: [string, any]) => ({
    name: key === 'NUPL' ? 'NUPL (Unrealized PnL)' : key === 'PM_proxy' ? 'Puell Multiple (Górnicy)' : 'Adresy Aktywne',
    rawKey: key,
    multiplier: typeof val?.coef === 'number' ? val.coef : 0,
    p_value: val?.p_value
  }));

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <Layers className="w-6 h-6 text-blue-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">ARDL & Kointegracja Pesaran Bounds V2 (Zero Look-Ahead)</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Artykuł bada długoterminową i krótkoterminową dynamikę pomiędzy wskaźnikami on-chain (takimi jak <span className="text-amber-400 font-semibold">NUPL</span> czy aktywność sieci) a ceną Bitcoina. Wersja V2 wykonuje rygorystyczny <span className="text-blue-400 font-mono">ARDL Bounds Test Pesarana</span>, korygując opóźnienia optymalnym kryterium AIC, co pozwala udowodnić istnienie stochastycznej kointegracji (wspólnego trendu długookresowego).
            </p>
          </div>
        </div>
      </div>

      {/* 1. Kafelki informacyjne kointegracji i ECM */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        
        {/* Bounds Test Tile */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Test Kointegracji (Bounds Test)</span>
            <div className="text-4xl font-bold font-mono text-white mt-2">
              F = {bounds_test.F_stat?.toFixed(2)}
            </div>
            <p className="text-slate-400 text-xs mt-2 mb-4">I(1) Bound przy 5% wynosi {bounds_test.I1_bound_5pct}.</p>
            <div className="mt-4 pt-3 border-t border-slate-800/60 text-xs font-bold text-slate-300">
              Werdykt: <span className={bounds_test.cointegration ? 'text-emerald-400' : 'text-amber-400'}>{bounds_test.decision}</span>
            </div>
          </div>
        </div>

        {/* ECM ECT Tile */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">Szybkość powrotu do równowagi (ECT)</span>
            <div className="text-4xl font-bold font-mono text-white mt-2">
              {ecm_results.ECT?.coef?.toFixed(3)}
            </div>
            <p className="text-slate-400 text-xs mt-2 mb-4">Error Correction Term. Znak minus wskazuje na prawidłową korektę błędu do równowagi długoterminowej.</p>
            <div className="mt-4 pt-3 border-t border-slate-800/60 text-xs font-bold text-slate-300">
              P-value: <span className={ecm_results.ECT?.p_value < 0.05 ? 'text-emerald-400' : 'text-red-400'}>{ecm_results.ECT?.p_value?.toExponential(2)}</span>
            </div>
          </div>
        </div>
        
        {/* ADF Test Tile */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-amber-400 uppercase tracking-wider">Diagnostyka stacjonarności (ADF/KPSS)</span>
            <div className="text-sm font-bold text-slate-200 mt-2 mb-2">
              LogPrice: {adf_tests.log_PriceUSD?.ADF_pvalue > 0.05 ? <span className="text-amber-400">Niestacjonarny I(1)</span> : <span className="text-emerald-400">Stacjonarny I(0)</span>}
            </div>
            <div className="text-sm font-bold text-slate-200 mb-2">
              NUPL: {adf_tests.NUPL?.ADF_pvalue > 0.05 ? <span className="text-amber-400">Niestacjonarny I(1)</span> : <span className="text-emerald-400">Stacjonarny I(0)</span>}
            </div>
            <p className="text-slate-400 text-xs mt-2 mb-4">ARDL wymaga, by żadna zmienna nie była I(2).</p>
          </div>
        </div>

      </div>

      {/* 2. Mnożniki długoterminowe ARDL */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-purple-400 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Mnożniki Długoterminowe (Long-Run Multipliers)
        </h3>
        <p className="text-slate-400 text-sm mb-6">Wskazują jak bardzo i w jakim kierunku zmienne objaśniające (on-chain) kształtują długoterminową wycenę Bitcoina. Dodatni, statystycznie istotny współczynnik oznacza, że wzrost wskaźnika wspiera długoterminowy wzrost ceny.</p>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={ltmData} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" stroke="#64748b" />
              <YAxis type="category" dataKey="name" stroke="#cbd5e1" tick={{ fontSize: 12 }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} 
                formatter={(val: unknown, name: any, props: any) => [
                  `${Number(val).toFixed(2)} (p=${props.payload.p_value?.toExponential(2)})`, 
                  'Mnożnik'
                ]} 
              />
              <Bar dataKey="multiplier" radius={[0, 4, 4, 0]}>
                {ltmData.map((entry: any, idx: number) => (
                  <Cell key={`cell-${idx}`} fill={entry.p_value < 0.05 ? (entry.multiplier > 0 ? '#38bdf8' : '#f43f5e') : '#64748b'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
    </div>
  );
}
