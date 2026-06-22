import Link from 'next/link';
import { papers } from '@/data/papers';
import { BookOpen } from 'lucide-react';

export default function LiteraturaLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col md:flex-row min-h-screen bg-slate-950 font-sans">
      <aside className="w-full md:w-80 bg-slate-900 border-r border-slate-800 p-6 h-full flex-shrink-0 shadow-xl z-10">
        <h2 className="text-xl font-bold mb-6 text-white flex items-center gap-2">
          <BookOpen className="text-emerald-500 w-6 h-6" />
          Literatura
        </h2>
        <ul className="space-y-3">
          {papers.map((p, index) => (
            <li key={p.id}>
              <Link 
                href={`/literatura/${p.id}`} 
                className="block p-3 rounded-xl border border-slate-800 hover:border-emerald-500 hover:bg-slate-800/50 text-sm text-slate-300 font-medium transition-all group"
              >
                <div className="flex items-start gap-2">
                  <span className="text-emerald-500 font-bold">{index + 1}.</span>
                  <span className="group-hover:text-emerald-400 transition-colors line-clamp-2">{p.title}</span>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      </aside>
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="max-w-4xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  );
}
