// generate a random number between 1 and 100
const randomNumber = Math.floor(Math.random() * 100)+1;

// Function to check the user's guess
let attempts = 0;
let maxAttempts = 5;
let maxAtemptsInput = document.getElementById("maxAttemptsInput");
let saveMaxAttemptsBtn = document.getElementById("saveMaxAttempts");
let newGameBtn = document.getElementById("newGameBtn");

//function to save the selected maximum attempts values and hide the input field
function saveMaxAttemptsBtn1() {
    let maxAtempts = parseInt(maxAtemptsInput.value)
    // Validate the input value
    if (maxAtempts > 0) {
        maxAttemptsInput.disabled = true;
        //saveMaxAttemptsBtn.style.display = "none";
        //save the maximum attempts value to use during the game
        // you can store it in a global variable or wherever you prefer
        console.log("Maximum attempts saved: ", maxAttempts);
        localStorage.setItem("maxAttempts", maxAttempts);
        saveMaxAttemptsBtn.style.display.disabled = True;
    } else {
        alert("please enter a valid number of attempts.")
    }
}

function checkGuess() {
    // get the user's guess from the input field
    const guessInput = document.getElementById("guessInput");
    const guess = parseInt(guessInput.value);

    //Get the feedback element
    const feedback = document.getElementById("feedback");
    const attemptsDisplay = document.getElementById("attempts");

    // Increment the number of attempts
    attempts++;

    // Display the number of attempts

    attemptsDisplay.textContent = `Attempts: ${attempts}`

    

    //check if the guess is correct

    if (guess === randomNumber) {
        feedback.textContent = "Congratulations! You guessed the correct number!";

    } else if (guess < randomNumber) {
        feedback.textContent = "Too low! Try a higher number.";
    } else {
        feedback.textContent = "Too high! Try a lower number.";
    }
    
    // check if the user has reached the maximum number of attempts
    if (attempts >= maxAttempts){
       gameOver();
    }
    // clear the input field
    guessInput.value = "";
}

// function to start a new game
function newGame() {
    // Reset the number of attempts
    attempts = 0;
    maxAttemptsInput.disabled = false;
    saveMaxAttemptsBtn.style.display = "inline-block";
    // reset the feedback and attempts display
    document.getElementById("feedback").textContent = "";
    document.getElementById("attempts").textContent = "";

    // Generate a new random number
    randomNumber = Math.floor(Math.random() * 100) + 1;

}

// function to handle the end of the game

function gameOver() {
    // Display a message indicating the game is Over
    document.getElementById("feedback").textContent = `Game over! The correct number was ${randomNumber}.`;

    // Disable the input field and submit button
    document.getElementById("guessInput").disabled = True;
    submitGuessBtn.disabled = true;
    newGameBtn.style.display = "inline-block";
}