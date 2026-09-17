import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, Terminal, Network, BookOpen } from 'lucide-react';

export default function HomePage() {
  return <main className="mx-auto w-full max-w-6xl px-6 py-16 md:py-24">
    <div className="grid gap-12 lg:grid-cols-2 lg:items-center">
      <div>
        <p className="mb-5 text-sm font-mono uppercase tracking-widest text-fd-muted-foreground">rGPU / Product documentation</p>
        <h1 className="text-5xl font-semibold tracking-tight leading-tight md:text-6xl">Your code, here.<br /><span className="text-teal-700 dark:text-teal-400">Your GPU, anywhere.</span></h1>
        <p className="mt-6 max-w-lg text-lg leading-relaxed text-fd-muted-foreground">Keep Python on your laptop. Run PyTorch operations and hold tensors on a remote GPU, including from a Mac with no CUDA installation.</p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link href="/docs/quickstart/" className="inline-flex items-center gap-2 rounded-lg bg-fd-primary px-5 py-3 font-medium text-fd-primary-foreground">Start with PyTorch <ArrowRight size={18} /></Link>
          <Link href="/docs/" className="rounded-lg border px-5 py-3 font-medium hover:bg-fd-accent">Compare the two paths</Link>
        </div>
      </div>
      <div className="home-hero-carousel grid">
        <div className="home-hero-image col-start-1 row-start-1 grid overflow-hidden rounded-xl border bg-fd-card shadow-sm">
          <div className="flex items-center gap-2 border-b px-5 py-3 text-sm text-fd-muted-foreground"><Network size={16} /> Local code · remote GPU</div>
          <div className="flex items-center bg-gradient-to-br from-cyan-50/70 via-transparent to-indigo-50/70 p-4 dark:from-cyan-950/20 dark:to-indigo-950/20 md:p-6">
            <Image
              src="/rgpu-architecture.png"
              width={1400}
              height={700}
              priority
              unoptimized
              alt="A laptop sends tensor operations through an encrypted tunnel to a remote GPU server and receives results"
              className="h-auto w-full"
            />
          </div>
          <div className="border-t px-5 py-4 text-sm text-fd-muted-foreground">Python stays with you. Tensors stay on the GPU.</div>
        </div>
        <div className="home-hero-code col-start-1 row-start-1 grid overflow-hidden rounded-xl border bg-fd-card shadow-sm">
          <div className="flex items-center gap-2 border-b px-5 py-3 text-sm text-fd-muted-foreground"><Terminal size={16} /> train.py · local Python</div>
          <pre className="self-center overflow-x-auto p-6 text-sm leading-8"><code><span className="text-violet-700 dark:text-violet-300">import</span>{` rgpu, torch

x = torch.randn(`}<span className="text-amber-700 dark:text-amber-300">1024</span>{`, `}<span className="text-amber-700 dark:text-amber-300">1024</span>{`, device=`}<span className="text-teal-700 dark:text-teal-300">"rgpu"</span>{`)
y = (x @ x).relu().sum()
`}<span className="text-cyan-700 dark:text-cyan-300">print</span>{`(y.item())`}</code></pre>
          <div className="border-t px-5 py-4 text-sm text-fd-muted-foreground">Local Python → SSH tunnel → GPU server</div>
        </div>
      </div>
    </div>
    <div className="mt-20 grid gap-4 md:grid-cols-3">
      {[
        { href: '/docs/training/', icon: BookOpen, title: 'Train with PyTorch', text: 'Move a model, run autograd, and ship compiled graphs.' },
        { href: '/docs/cuda-shim/', icon: Network, title: 'Use the CUDA shim', text: 'Keep device="cuda" with the Linux driver-remoting path.' },
        { href: '/docs/performance/', icon: Terminal, title: 'Understand the cost', text: 'Measure host waits, transfers, and completed GPU work.' },
      ].map(({ href, icon: Icon, title, text }) => <Link key={href} href={href} className="rounded-xl border p-6 transition-colors hover:bg-fd-accent"><Icon size={22} className="mb-5 text-teal-700 dark:text-teal-400" /><h2 className="font-semibold">{title}</h2><p className="mt-2 text-sm leading-relaxed text-fd-muted-foreground">{text}</p></Link>)}
    </div>
    <p className="mt-10 text-sm text-fd-muted-foreground">Built for trusted GPU hosts. <Link href="/docs/operations/" className="underline underline-offset-4">Read the deployment guidance</Link> before connecting.</p>
  </main>;
}
