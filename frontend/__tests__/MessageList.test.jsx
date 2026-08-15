import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import MessageList from '../src/MessageList.jsx';

const baseProps = {
  sessionId: 'sess_1',
  onFeedback: jest.fn(),
  feedbackState: {},
};

describe('MessageList', () => {
  it('shows placeholder when no messages', () => {
    render(<MessageList messages={[]} {...baseProps} />);
    expect(screen.getByText(/Ask a question about FastAPI/)).toBeInTheDocument();
  });

  it('renders user and assistant messages', () => {
    const messages = [
      { id: '1', role: 'user', content: 'Hello' },
      { id: '2', role: 'assistant', content: 'Hi there', streaming: false, citations: [], confidence: 0.9 },
    ];
    render(<MessageList messages={messages} {...baseProps} />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
    expect(screen.getByText('Hi there')).toBeInTheDocument();
  });

  it('renders citation chips for assistant messages with citations', () => {
    const messages = [
      {
        id: '1',
        role: 'assistant',
        content: 'See the docs',
        streaming: false,
        citations: [{ title: 'FastAPI', source_url: 'https://fastapi.tiangolo.com' }],
        confidence: 0.9,
      },
    ];
    render(<MessageList messages={messages} {...baseProps} />);
    expect(screen.getByText('[Source: FastAPI]')).toBeInTheDocument();
  });

  it('shows low-confidence warning when confidence < 0.65', () => {
    const messages = [
      {
        id: '1',
        role: 'assistant',
        content: 'Maybe',
        streaming: false,
        citations: [],
        confidence: 0.3,
      },
    ];
    render(<MessageList messages={messages} {...baseProps} />);
    expect(screen.getByText(/Low confidence/)).toBeInTheDocument();
  });

  it('does not show warning when confidence >= 0.65', () => {
    const messages = [
      {
        id: '1',
        role: 'assistant',
        content: 'Definitely',
        streaming: false,
        citations: [],
        confidence: 0.8,
      },
    ];
    render(<MessageList messages={messages} {...baseProps} />);
    expect(screen.queryByText(/Low confidence/)).not.toBeInTheDocument();
  });

  it('shows feedback buttons for completed assistant messages', () => {
    const messages = [
      { id: '1', role: 'assistant', content: 'Answer', streaming: false, citations: [], confidence: 0.9 },
    ];
    render(<MessageList messages={messages} {...baseProps} />);
    expect(screen.getByTestId('feedback-up-0')).toBeInTheDocument();
    expect(screen.getByTestId('feedback-down-0')).toBeInTheDocument();
  });

  it('hides feedback buttons while streaming', () => {
    const messages = [
      { id: '1', role: 'assistant', content: '', streaming: true, citations: [], confidence: null },
    ];
    render(<MessageList messages={messages} {...baseProps} />);
    expect(screen.queryByTestId('feedback-up-0')).not.toBeInTheDocument();
  });

  it('calls onFeedback with message index and rating', () => {
    const onFeedback = jest.fn();
    const messages = [
      { id: '1', role: 'assistant', content: 'Answer', streaming: false, citations: [], confidence: 0.9 },
    ];
    render(<MessageList messages={messages} sessionId="sess_1" onFeedback={onFeedback} feedbackState={{}} />);
    fireEvent.click(screen.getByTestId('feedback-up-0'));
    expect(onFeedback).toHaveBeenCalledWith(0, 'up');
  });

  it('disables feedback buttons when no sessionId', () => {
    const messages = [
      { id: '1', role: 'assistant', content: 'Answer', streaming: false, citations: [], confidence: 0.9 },
    ];
    render(<MessageList messages={messages} sessionId={null} onFeedback={jest.fn()} feedbackState={{}} />);
    expect(screen.getByTestId('feedback-up-0')).toBeDisabled();
  });

  it('shows error text instead of feedback buttons on error', () => {
    const messages = [
      { id: '1', role: 'assistant', content: '', streaming: false, error: 'Network failed', citations: [], confidence: null },
    ];
    render(<MessageList messages={messages} {...baseProps} />);
    expect(screen.getByText('Network failed')).toBeInTheDocument();
    expect(screen.queryByTestId('feedback-up-0')).not.toBeInTheDocument();
  });

  it('marks feedback button as active when feedbackState is set', () => {
    const messages = [
      { id: '1', role: 'assistant', content: 'Answer', streaming: false, citations: [], confidence: 0.9 },
    ];
    render(
      <MessageList
        messages={messages}
        sessionId="sess_1"
        onFeedback={jest.fn()}
        feedbackState={{ 0: 'up' }}
      />,
    );
    expect(screen.getByTestId('feedback-up-0')).toHaveClass('feedback-btn--active');
  });
});
