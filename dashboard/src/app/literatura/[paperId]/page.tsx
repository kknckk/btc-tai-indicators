'use client';

import { useParams } from 'next/navigation';
import { papers } from '@/data/papers';
import { BookOpen, Code, FlaskConical, Link as LinkIcon, AlertCircle } from 'lucide-react';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import dynamic from 'next/dynamic';

const Paper1Dashboard = dynamic(() => import('@/components/literature/Paper1Dashboard'), { ssr: false, loading: () => <p>Ładowanie wyników analitycznych...</p> });
const Paper2Dashboard = dynamic(() => import('@/components/literature/Paper2Dashboard'), { ssr: false, loading: () => <p>Ładowanie wyników analitycznych...</p> });
const Paper3Dashboard = dynamic(() => import('@/components/literature/Paper3Dashboard'), { ssr: false, loading: () => <p>Ładowanie wyników analitycznych...</p> });
const Paper4Dashboard = dynamic(() => import('@/components/literature/Paper4Dashboard'), { ssr: false, loading: () => <p>Ładowanie wyników analitycznych...</p> });
const Paper5Dashboard = dynamic(() => import('@/components/literature/Paper5Dashboard'), { ssr: false, loading: () => <p>Ładowanie wyników analitycznych...</p> });
const Paper6Dashboard = dynamic(() => import('@/components/literature/Paper6Dashboard'), { ssr: false, loading: () => <p>Ładowanie wyników analitycznych...</p> });
const Paper7Dashboard = dynamic(() => import('@/components/literature/Paper7Dashboard'), { ssr: false, loading: () => <p>Ładowanie wyników analitycznych...</p> });
const Paper8Dashboard = dynamic(() => import('@/components/literature/Paper8Dashboard'), { ssr: false, loading: () => <p>Ładowanie wyników analitycznych...</p> });
const Paper9Dashboard = dynamic(() => import('@/components/literature/Paper9Dashboard'), { ssr: false, loading: () => <p>Ładowanie wyników analitycznych...</p> });
const Paper10Dashboard = dynamic(() => import('@/components/literature/Paper10Dashboard'), { ssr: false, loading: () => <p>Ładowanie wyników analitycznych...</p> });

export default function PaperPage() {
  const { paperId } = useParams();
  const idStr = typeof paperId === 'string' ? paperId : (paperId?.[0] || '');
  const details = papers.find(p => p.id === idStr);

  if (!details) {
    return (
      <div className="p-8 text-center bg-slate-900 border border-slate-800 rounded-2xl shadow-xl">
        <AlertCircle className="w-12 h-12 text-slate-500 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-white mb-2">Brak artykułu</h2>
        <p className="text-slate-400">Artykuł o ID &quot;{idStr}&quot; nie istnieje w bazie lub wdrożenie jest w toku.</p>
      </div>
    );
  }

  return (
    <div className="max-w-5xl space-y-8 font-sans">
      <header>
        <h1 className="text-3xl font-extrabold text-white mb-4 leading-tight">{details.title}</h1>
        <a 
          href={details.url} 
          target="_blank" 
          rel="noreferrer"
          className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors"
        >
          <LinkIcon className="w-4 h-4" />
          <span className="font-medium">Link do publikacji</span>
        </a>
      </header>
      
      <div className="grid grid-cols-1 gap-6">
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
          <h3 className="font-bold text-xl text-emerald-400 mb-3 flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            Skrót (Teza badawcza)
          </h3>
          <p className="text-slate-300 leading-relaxed">{details.summary}</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
          <h3 className="font-bold text-xl text-blue-400 mb-3 flex items-center gap-2">
            <FlaskConical className="w-5 h-5" />
            Jak odtworzyć wyniki (Metodologia)
          </h3>
          <p className="text-slate-300 leading-relaxed mb-4">{details.reproductionSteps}</p>
          <div className="flex flex-wrap gap-2">
            {details.techStack.map(tech => (
              <span key={tech} className="bg-blue-900/30 text-blue-300 border border-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                {tech}
              </span>
            ))}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
          <h3 className="font-bold text-xl text-amber-400 mb-3 flex items-center gap-2">
            <Code className="w-5 h-5" />
            Dostępność kodu (Weryfikacja)
          </h3>
          <p className="text-slate-300 leading-relaxed">{details.codeAvailability}</p>
        </div>
      </div>

      <div className="mt-12 p-8 border border-slate-800 rounded-2xl bg-slate-900 shadow-xl">
        <div className="prose prose-invert prose-emerald max-w-none">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]}
            components={{
              h3: ({...props}: any) => <h3 className="text-2xl font-bold text-white mb-6 border-b border-slate-800 pb-2 flex items-center gap-2" {...props} />,
              p: ({...props}: any) => <p className="text-slate-300 mb-4 leading-relaxed" {...props} />,
              ol: ({...props}: any) => <ol className="list-decimal list-inside space-y-4 text-slate-300 ml-2" {...props} />,
              li: ({...props}: any) => <li className="pl-2 marker:text-emerald-500 marker:font-bold" {...props} />,
              strong: ({...props}: any) => <strong className="text-emerald-400 font-semibold" {...props} />,
              code: ({inline, ...props}: any) => 
                inline 
                  ? <code className="bg-slate-800 text-blue-300 px-1.5 py-0.5 rounded font-mono text-sm" {...props} />
                  : <code className="block bg-slate-950 p-4 rounded-xl text-sm overflow-x-auto border border-slate-800 text-slate-300 my-4" {...props} />
            }}
          >
            {details.implementationPlan}
          </ReactMarkdown>
        </div>
      </div>
      
      {/* Sekcja Wyników Analitycznych (Dashboardy) */}
      <div className="mt-12">
        <h2 className="text-3xl font-extrabold text-white mb-6">Wyniki Analityczne (Live)</h2>
        {idStr === 'paper1' && <Paper1Dashboard />}
        {idStr === 'paper2' && <Paper2Dashboard />}
        {idStr === 'paper3' && <Paper3Dashboard />}
        {idStr === 'paper4' && <Paper4Dashboard />}
        {idStr === 'paper5' && <Paper5Dashboard />}
        {idStr === 'paper6' && <Paper6Dashboard />}
        {idStr === 'paper7' && <Paper7Dashboard />}
        {idStr === 'paper8' && <Paper8Dashboard />}
        {idStr === 'paper9' && <Paper9Dashboard />}
        {idStr === 'paper10' && <Paper10Dashboard />}
      </div>
    </div>
  );
}
