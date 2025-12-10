import React from 'react';

function SignInButton({label}) {
  const handleSignInClick = (e) => {
    e.preventDefault();
    if (window.openSignInModal) {
      window.openSignInModal();
    }
  };

  return (
    <a
      className="navbar__link"
      href="#"
      onClick={handleSignInClick}
    >
      {label || 'Sign In'}
    </a>
  );
}

export default SignInButton;