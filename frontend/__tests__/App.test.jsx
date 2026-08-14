import React from 'react';
import { render, screen } from '@testing-library/react';
import App from '../src/App.jsx';

describe('App scaffold', () => {
  it('renders the project title', () => {
    render(<App />);
    expect(screen.getByText('CTC-RAG Chat Assistant')).toBeInTheDocument();
  });
});
