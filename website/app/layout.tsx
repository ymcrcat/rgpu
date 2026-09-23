import type { Metadata } from 'next';
import { Provider } from '@/components/provider';
import './global.css';

const title = 'rGPU documentation';
const description =
  'Run PyTorch from your laptop on a remote GPU. Setup, training, CUDA remoting, and practical performance guidance.';

export const metadata: Metadata = {
  // Absolute URLs are required by crawlers: without metadataBase, Next emits a
  // relative og:image and Twitter renders no card at all.
  metadataBase: new URL('https://rgpu.dev'),
  title: { default: title, template: '%s | rGPU' },
  description,
  openGraph: {
    type: 'website',
    siteName: 'rGPU',
    url: 'https://rgpu.dev',
    title,
    description,
    images: ['/og.png'],
  },
  twitter: {
    card: 'summary_large_image',
    title,
    description,
    images: ['/og.png'],
  },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return <html lang="en" suppressHydrationWarning><body className="flex flex-col min-h-screen"><Provider>{children}</Provider></body></html>;
}
