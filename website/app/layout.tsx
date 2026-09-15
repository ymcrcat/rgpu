import type { Metadata } from 'next';
import { Provider } from '@/components/provider';
import './global.css';

export const metadata: Metadata = {
  title: { default: 'rGPU documentation', template: '%s | rGPU' },
  description: 'Run PyTorch from your laptop on a remote GPU. Setup, training, CUDA remoting, and practical performance guidance.',
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return <html lang="en" suppressHydrationWarning><body className="flex flex-col min-h-screen"><Provider>{children}</Provider></body></html>;
}
