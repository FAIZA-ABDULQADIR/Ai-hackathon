import React, { useState, useCallback } from 'react';
import clsx from 'clsx';
import { useLocation } from '@docusaurus/router';
import Link from '@docusaurus/Link';
import { isRegexpStringMatch } from '@docusaurus/theme-common';
import SignInModal from '@site/src/components/SignInModal';

function SignInButtonNavbarItem({
  mobile,
  position,
  label = 'Sign In',
  to,
  href,
  ...props
}) {
  const [showSignInModal, setShowSignInModal] = useState(false);
  const location = useLocation();

  const openSignInModal = useCallback(() => {
    setShowSignInModal(true);
  }, []);

  const closeSignInModal = useCallback(() => {
    setShowSignInModal(false);
  }, []);

  const isActive = href
    ? isRegexpStringMatch(href, location.pathname)
    : to
    ? isRegexpStringMatch(to, location.pathname)
    : false;

  const navbarItemClasses = clsx(
    'navbar__item navbar__link',
    {
      'navbar__link--active': isActive,
    },
    props.className
  );

  return (
    <>
      <Link
        className={navbarItemClasses}
        onClick={(e) => {
          e.preventDefault();
          openSignInModal();
        }}
        {...(href && { href })}
        {...(to && { to })}
      >
        {label}
      </Link>
      <SignInModal
        isOpen={showSignInModal}
        onClose={closeSignInModal}
      />
    </>
  );
}

export default SignInButtonNavbarItem;