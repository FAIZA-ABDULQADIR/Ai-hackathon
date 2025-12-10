import React from 'react';
import NavbarItem from '@theme/NavbarItem';
import SignInButton from '@site/src/components/SignInButton';

function CustomSignInButtonNavbarItem({
  mobile,
  position,
  label,
  ...props
}) {
  return (
    <div className="navbar__item">
      <SignInButton />
    </div>
  );
}

export default CustomSignInButtonNavbarItem;