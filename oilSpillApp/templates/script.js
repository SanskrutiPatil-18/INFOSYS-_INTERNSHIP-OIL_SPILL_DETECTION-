// Navbar scroll effects
function handleNavbarScroll() {
  const navbar = document.querySelector(".navbar");
  if (!navbar) return;
  navbar.style.background = "#064e7b";
  navbar.style.boxShadow =
    window.scrollY > 100 ? "0 2px 20px rgba(0, 0, 0, 0.1)" : "none";
}

window.addEventListener("scroll", handleNavbarScroll);

// Hero content animation
document.addEventListener("DOMContentLoaded", () => {
  const heroContent = document.querySelector(".hero-content");
  if (heroContent) {
    heroContent.style.opacity = "0";
    heroContent.style.transform = "translateY(30px)";
    setTimeout(() => {
      heroContent.style.transition = "opacity 1s ease, transform 1s ease";
      heroContent.style.opacity = "1";
      heroContent.style.transform = "translateY(0)";
    }, 100);
  }

// Feature cards: flip on click and center with backdrop blur
  const featureCards = document.querySelectorAll(".features .feature-card");
  const backdrop = document.getElementById("feature-backdrop");
  if (!featureCards.length || !backdrop) return;

  const openCard = (card) => {
    document.querySelectorAll(".feature-card.active").forEach((c) => {
      c.classList.remove("active");
      c.style.opacity = "1";
      const innerPrev = c.querySelector(".card-inner");
      if (innerPrev) {
        innerPrev.classList.remove("is-flipped");
        innerPrev.style.transform = "";
      }
      const frontPrev = c.querySelector(".card-front");
      const backPrev = c.querySelector(".card-back");
      if (frontPrev) {
        frontPrev.style.opacity = "1";
        frontPrev.style.visibility = "visible";
        frontPrev.style.transform = "";
      }
      if (backPrev) {
        backPrev.style.opacity = "0";
        backPrev.style.visibility = "hidden";
        backPrev.style.transform = "";
      }
    });
    card.style.transform = "";
    card.style.opacity = "1";
    card.classList.add("active");
    const inner = card.querySelector(".card-inner");
    if (inner) inner.classList.add("is-flipped");
    backdrop.classList.add("show");
    document.body.style.overflow = "hidden";
  };

  const closeActive = () => {
    document.querySelectorAll(".feature-card.active").forEach((c) => {
      c.classList.remove("active");
      c.style.opacity = "1";
      const inner = c.querySelector(".card-inner");
      if (inner) {
        inner.classList.remove("is-flipped");
        inner.style.transform = "";
      }
      const front = c.querySelector(".card-front");
      const back = c.querySelector(".card-back");
      if (front) {
        front.style.opacity = "1";
        front.style.visibility = "visible";
        front.style.transform = "";
      }
      if (back) {
        back.style.opacity = "0";
        back.style.visibility = "hidden";
        back.style.transform = "";
      }
    });
    backdrop.classList.remove("show");
    document.body.style.overflow = "";
  };

  featureCards.forEach((card) => {
    card.addEventListener("click", function () {
      if (!this.classList.contains("active")) openCard(this);
    });
    card.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (!this.classList.contains("active")) openCard(this);
      }
    });
  });

  backdrop.addEventListener("click", closeActive);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeActive();
  });
});
