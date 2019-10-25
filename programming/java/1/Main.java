import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;

public class Main {

    public static void main(String[] args) {

        ArrayList<Ball> ballList = new ArrayList<>(Arrays.asList(new Ball("2.33"),
                new Ball("1"), new Ball(4.)));

        System.out.println("Balls: ");
        for (Ball ball : ballList) {
            System.out.println(ball);
        }
        System.out.println("\nVolume and Surface area of 1st ball: " + ballList.get(0).Volume() +
                ' ' +
                ballList.get(0).SurfaceArea());

        Collections.sort(ballList);
        System.out.println("\nSorted by radius: ");
        for (Ball ball : ballList) {
            System.out.println(ball);
        }

        Ball.setSorting_criterion("sortByID");
        Collections.sort(ballList);
        System.out.println("\nSorted by ID: ");
        for (Ball ball : ballList) {
            System.out.println(ball);
        }

        System.out.println("\nOutput of balls' fields through iterator: ");
        for (Ball ball : ballList) {
            Iterator<Object> iterator = ball.iterator();
            while (iterator.hasNext()) {
                System.out.print(iterator.next() + " ");
            }
            System.out.print("\n");
        }


    }
}
