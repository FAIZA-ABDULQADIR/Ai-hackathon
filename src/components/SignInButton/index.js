import React, { useState, useCallback } from 'react';
import SignInModal from '../SignInModal';

// This component will be used to wrap the sign in functionality
export default function SignInButton({ children }) {
  const [showSignInModal, setShowSignInModal] = useState(false);

  const openSignInModal = useCallback(() => {
    setShowSignInModal(true);
  }, []);

  const closeSignInModal = useCallback(() => {
    setShowSignInModal(false);
  }, []);

  return (
    <>
      <div onClick={openSignInModal} style={{ cursor: 'pointer' }}>
        {children}
      </div>
      <SignInModal
        isOpen={showSignInModal}
        onClose={closeSignInModal}
      />
    </>
  );
}