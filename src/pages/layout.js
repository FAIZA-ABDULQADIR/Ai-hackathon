import React from 'react';
import GlobalSignInHandler from '../components/GlobalSignInHandler';

export default function Layout({children}) {
  return (
    <>
      {children}
      <GlobalSignInHandler />
    </>
  );
}