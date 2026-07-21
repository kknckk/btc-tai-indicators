'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ScatterChart, Scatter, ReferenceLine
} from 'recharts';
import { ShieldAlert, CheckCircle, Activity, Layers, Cpu, Award } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper10Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper10', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie Systemu Wczesnego Ostrzegania (PCA & Regresja Logistyczna)...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 10.</div>;

  const { explained_variance = [], classification_metrics = {}, time_series = [], scatter_data = [] } = data;

  const evData = explained_variance.map((val: number, idx: number) => ({
    pc: `Główna Składowa PC${idx + 1}`,
    variance: val * 100
  }));

  const cm = classification_metrics.confusion_matrix ?? [[0, 0], [0, 0]];

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <ShieldAlert className="w-6 h-6 text-red-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">System Wczesnego Ostrzegania przed Załamaniami Rynku (PCA & Regresja Logistyczna)</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Konstrukcja wieloczynnikowego wskaźnika ryzyka (Early Warning Indicator - EWI) poprzez redukcję wymiarowości wielkich zbiorów danych on-chain za pomocą <span className="text-blue-400 font-semibold">Analizy Głównych Składowych (PCA)</span>. Pierwsze składowe <span className="font-mono text-xs">PC1</span> i <span className="font-mono text-xs">PC2</span> zasilają regresję logistyczną przewidującą wejście rynku w reżim skrajnie wysokiej zmienności 30-dniowej (<span className="text-red-400 font-semibold">rv_30d &gt; 90. percentyla</span>).
            </p>
          </div>
        </div>
      </div>

      {/* 1. Kafelki metryk wczesnego ostrzegania i tabela PCA */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Metryki EWI */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-red-400 uppercase tracking-wider">Skuteczność Alarmowania (Accuracy & Precision)</span>
            <div className="text-3xl font-bold font-mono text-white mt-2">
              {((classification_metrics.accuracy ?? 0) * 100).toFixed(1)}% <span className="text-xs font-normal text-slate-400">Accuracy</span>
            </div>
            <div className="grid grid-cols-2 gap-3 mt-4 text-xs text-slate-300 border-t border-slate-800 pt-3">
              <div>Precision: <span className="font-mono font-bold text-emerald-400">{((classification_metrics.precision ?? 0) * 100).toFixed(1)}%</span></div>
              <div>Recall: <span className="font-mono font-bold text-blue-400">{((classification_metrics.recall ?? 0) * 100).toFixed(1)}%</span></div>
            </div>
            {/* Macierz pomyłek */}
            <div className="mt-4 pt-3 border-t border-slate-800/60 text-xs">
              <div className="text-slate-400 mb-2">Macierz pomyłek EWI (Confusion Matrix):</div>
              <div className="grid grid-cols-2 gap-2 text-center font-mono">
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-emerald-400">TN: {cm[0][0]}</div>
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-red-400">FP (Fałszywy alarm): {cm[0][1]}</div>
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-amber-400">FN (Przegapienie): {cm[1][0]}</div>
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-emerald-400 font-bold">TP (Trafny alarm): {cm[1][1]}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Wykres wariancji PCA */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg lg:col-span-2 flex flex-col justify-between">
          <div>
            <h3 className="font-bold text-lg text-blue-400 mb-3 flex items-center gap-2">
              <Layers className="w-5 h-5" />
              Wariancja Wyjaśniona przez Składowe On-Chain (PCA Explained Variance)
            </h3>
            <p className="text-slate-400 text-xs mb-4">Skupienie wariancji wielowymiarowych wskaźników sieciowych w pierwszych ortogonalnych wektorach własnych.</p>
            <div className="h-[220px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={evData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="pc" stroke="#64748b" />
                  <YAxis stroke="#94a3b8" tickFormatter={(v) => `${v.toFixed(0)}%`} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} formatter={(v: any) => [`${Number(v).toFixed(2)}%`, 'Wariancja wyjaśniona']} />
                  <Bar dataKey="variance" radius={[6, 6, 0, 0]}>
                    {evData.map((entry: any, idx: number) => (
                      <Cell key={`cell-${idx}`} fill={idx === 0 ? '#38bdf8' : '#818cf8'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Wykres Prawdopodobieństwa EWI vs Zmienność i Cena */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-red-400 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Prawdopodobieństwo Alarmowe (EWI Probability) a Zmienność 30-dniowa
        </h3>
        <p className="text-slate-400 text-sm mb-6">Linia czerwona przedstawia wygenerowane przez system prawdopodobieństwo wejścia w kryzys zmienności (<span className="text-red-400 font-mono">ewi_prob</span>).</p>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={time_series} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('T')[0].split('-')[0]} />
              <YAxis yAxisId="left" stroke="#3b82f6" domain={['auto', 'auto']} tickFormatter={(val) => val.toFixed(2)} />
              <YAxis yAxisId="right" orientation="right" stroke="#ef4444" domain={[0, 1]} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} formatter={(val: any, name: any) => [name === 'Prawdopodobieństwo EWI' ? `${(Number(val)*100).toFixed(1)}%` : Number(val).toFixed(4), name]} />
              <ReferenceLine yAxisId="right" y={0.5} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'Próg alarmowy 50%', fill: '#ef4444', position: 'insideTopRight' }} />
              <Line yAxisId="left" type="monotone" dataKey="rv_30d" stroke="#38bdf8" dot={false} strokeWidth={2} name="Zmienność 30d (rv_30d)" />
              <Line yAxisId="right" type="monotone" dataKey="ewi_prob" stroke="#ef4444" dot={false} strokeWidth={2} name="Prawdopodobieństwo EWI" opacity={0.9} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Wykres rozrzutu PC1 vs PC2 z zaznaczeniem reżimu wysokiej zmienności */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-blue-400 mb-4 flex items-center gap-2">
          <Cpu className="w-5 h-5" />
          Przestrzeń Głównych Składowych (PC1 vs PC2) a Reżim Zmienności
        </h3>
        <p className="text-slate-400 text-sm mb-6">Rozdzielenie dni normalnego handlu (niebieskie) od skrajnie wysokiej zmienności (czerwone) w przestrzeni skróconej PCA.</p>
        <div className="h-[340px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" dataKey="PC1" name="PC1" stroke="#64748b" tickFormatter={(v) => v.toFixed(1)} />
              <YAxis type="number" dataKey="PC2" name="PC2" stroke="#64748b" tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }} />
              <Scatter name="Normalna Zmienność" data={scatter_data.filter((d: any) => !d.is_high_vol)} fill="#38bdf8" fillOpacity={0.5} />
              <Scatter name="Wysoka Zmienność (Top 10%)" data={scatter_data.filter((d: any) => d.is_high_vol)} fill="#ef4444" fillOpacity={0.85} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
