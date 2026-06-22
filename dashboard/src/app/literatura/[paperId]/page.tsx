'use client';

import { useParams } from 'next/navigation';
import { papers } from '@/data/papers';
import { BookOpen, Code, FlaskConical, Link as LinkIcon, AlertCircle } from 'lucide-react';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function PaperPage() {
  const { paperId } = useParams();
  const idStr = typeof paperId === 'string' ? paperId : (paperId?.[0] || '');
  const details = papers.find(p => p.id === idStr);

  if (!details) {
    return (
      <div className="p-8 text-center bg-slate-900 border border-slate-800 rounded-2xl shadow-xl">
        <AlertCircle className="w-12 h-12 text-slate-500 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-white mb-2">Brak artykułu</h2>
        <p className="text-slate-400">Artykuł o ID "{idStr}" nie istnieje w bazie lub wdrożenie jest w toku.</p>
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
              h3: ({node, ...props}) => <h3 className="text-2xl font-bold text-white mb-6 border-b border-slate-800 pb-2 flex items-center gap-2" {...props} />,
              p: ({node, ...props}) => <p className="text-slate-300 mb-4 leading-relaxed" {...props} />,
              ol: ({node, ...props}) => <ol className="list-decimal list-inside space-y-4 text-slate-300 ml-2" {...props} />,
              li: ({node, ...props}) => <li className="pl-2 marker:text-emerald-500 marker:font-bold" {...props} />,
              strong: ({node, ...props}) => <strong className="text-emerald-400 font-semibold" {...props} />,
              code: ({node, inline, ...props}: any) => 
                inline 
                  ? <code className="bg-slate-800 text-blue-300 px-1.5 py-0.5 rounded font-mono text-sm" {...props} />
                  : <code className="block bg-slate-950 p-4 rounded-xl text-sm overflow-x-auto border border-slate-800 text-slate-300 my-4" {...props} />
            }}
          >
            {details.implementationPlan}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
