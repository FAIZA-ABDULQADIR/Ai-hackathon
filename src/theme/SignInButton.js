import React from 'react';
import clsx from 'clsx';

function SignInButton({label}) {
  const handleSignInClick = (e) => {
    e.preventDefault();
    if (window.openSignInModal) {
      window.openSignInModal();
    }
  };

  return (
    <a
      className={clsx('navbar__item navbar__link', 'button button--secondary')}
      href="#"
      onClick={handleSignInClick}
    >
      {label || 'Sign In'}
    </a>
  );
}

export default SignInButton;