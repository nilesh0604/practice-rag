import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import InputBox from '../src/InputBox.jsx';

describe('InputBox', () => {
  it('renders a textarea and send button', () => {
    render(<InputBox onSend={jest.fn()} disabled={false} />);
    expect(screen.getByTestId('chat-input')).toBeInTheDocument();
    expect(screen.getByTestId('send-button')).toBeInTheDocument();
  });

  it('calls onSend with trimmed text and clears the input', () => {
    const onSend = jest.fn();
    render(<InputBox onSend={onSend} disabled={false} />);
    const input = screen.getByTestId('chat-input');
    fireEvent.change(input, { target: { value: '  hello  ' } });
    fireEvent.click(screen.getByTestId('send-button'));
    expect(onSend).toHaveBeenCalledWith('hello');
    expect(input.value).toBe('');
  });

  it('does not send empty or whitespace-only text', () => {
    const onSend = jest.fn();
    render(<InputBox onSend={onSend} disabled={false} />);
    fireEvent.change(screen.getByTestId('chat-input'), { target: { value: '   ' } });
    fireEvent.click(screen.getByTestId('send-button'));
    expect(onSend).not.toHaveBeenCalled();
  });

  it('sends on Enter (without Shift)', () => {
    const onSend = jest.fn();
    render(<InputBox onSend={onSend} disabled={false} />);
    const input = screen.getByTestId('chat-input');
    fireEvent.change(input, { target: { value: 'test' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: false });
    expect(onSend).toHaveBeenCalledWith('test');
  });

  it('does not send on Shift+Enter (inserts newline)', () => {
    const onSend = jest.fn();
    render(<InputBox onSend={onSend} disabled={false} />);
    const input = screen.getByTestId('chat-input');
    fireEvent.change(input, { target: { value: 'test' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();
  });

  it('disables textarea and button when disabled prop is true', () => {
    render(<InputBox onSend={jest.fn()} disabled={true} />);
    expect(screen.getByTestId('chat-input')).toBeDisabled();
    expect(screen.getByTestId('send-button')).toBeDisabled();
  });

  it('disables send button when text is empty', () => {
    render(<InputBox onSend={jest.fn()} disabled={false} />);
    expect(screen.getByTestId('send-button')).toBeDisabled();
  });
});
