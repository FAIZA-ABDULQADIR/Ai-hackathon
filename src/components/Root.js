import React from 'react';
import GlobalSignInHandler from './GlobalSignInHandler';

export default function Root({ children }) {
  return (
    <>
      {children}
      <GlobalSignInHandler />
    </>
  );
}