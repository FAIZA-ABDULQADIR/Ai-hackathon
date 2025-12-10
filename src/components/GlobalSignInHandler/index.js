import { useEffect } from 'react';
import SignInModal from '../SignInModal';
import { useState } from 'react';

// Global component that handles sign-in modal functionality
export default function GlobalSignInHandler() {
  const [showSignInModal, setShowSignInModal] = useState(false);

  useEffect(() => {
    // Add click handler to sign in links
    const handleSignInClick = (event) => {
      // Check if the clicked element or its parent has the sign in link
      const signInLink = event.target.closest('a[href="#"]');
      if (signInLink && signInLink.textContent.includes('Sign In')) {
        event.preventDefault();
        setShowSignInModal(true);
      }
    };

    // Add global click listener
    document.addEventListener('click', handleSignInClick);

    // Expose function globally for navbar item
    window.openSignInModal = () => setShowSignInModal(true);

    // Cleanup
    return () => {
      document.removeEventListener('click', handleSignInClick);
      delete window.openSignInModal;
    };
  }, []);

  const closeSignInModal = () => {
    setShowSignInModal(false);
  };

  return (
    <SignInModal
      isOpen={showSignInModal}
      onClose={closeSignInModal}
    />
  );
}