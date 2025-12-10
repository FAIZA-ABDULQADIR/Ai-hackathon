import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './SignInModal.module.css';

function SignInModal({ isOpen, onClose }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);

  if (!isOpen) {
    return null;
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    // In a real implementation, this would connect to Firebase/Auth0/Supabase
    console.log('Sign in attempt with:', { email, password, isSignUp });
    onClose();
  };

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>{isSignUp ? 'Sign Up' : 'Sign In'}</h2>
          <button className={styles.closeButton} onClick={onClose}>
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.signInForm}>
          <div className={styles.inputGroup}>
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className={styles.inputGroup}>
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <button type="submit" className={styles.submitButton}>
            {isSignUp ? 'Sign Up' : 'Sign In'}
          </button>
        </form>

        <div className={styles.modalFooter}>
          <p>
            {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
            <button
              className={styles.toggleModeButton}
              onClick={() => setIsSignUp(!isSignUp)}
            >
              {isSignUp ? 'Sign In' : 'Sign Up'}
            </button>
          </p>

          <div className={styles.divider}>
            <span>OR</span>
          </div>

          <button className={styles.googleButton}>
            Continue with Google
          </button>
        </div>
      </div>
    </div>
  );
}

export default SignInModal;