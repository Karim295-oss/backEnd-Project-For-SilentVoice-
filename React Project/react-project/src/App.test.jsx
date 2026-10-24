import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the translator screen with both feature tabs', () => {
  render(<App />);

  expect(screen.getByRole('heading', { name: /SILENTVOICE/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Sign to Text/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Speech to Sign/i })).toBeInTheDocument();
  expect(screen.getByRole('combobox', { name: /Recognition model/i })).toHaveValue('current');
  expect(screen.getByRole('option', { name: /ASL-100 TGCN/i })).toBeInTheDocument();
});
