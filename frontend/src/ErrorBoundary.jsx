import React from 'react';

/**
 * ErrorBoundary — catches render errors in the chat widget subtree and
 * shows a recovery UI instead of a blank screen.
 *
 * Per the architecture doc's component tree:
 *   App → ChatWidget → ErrorBoundary
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error('ErrorBoundary caught:', error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary" role="alert">
          <p>Something went wrong rendering the chat.</p>
          <p className="error-boundary__detail">
            {this.state.error?.message ?? 'Unknown error'}
          </p>
          <button className="btn" onClick={this.handleReset}>
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
